#include <hip/hip_runtime.h>
#include "gpu_backend.h"
#include <cfloat>
#include <cmath>

namespace digitrec {

// ---------------------------------------------------------------------------
// Kernel: matrix-vector multiply  output[i] = bias[i] + sum_j(weights[i*cols+j] * input[j])
// Each thread computes one output element.
// ---------------------------------------------------------------------------
__global__ void kernel_matvec(const double* __restrict__ weights,
                              const double* __restrict__ input,
                              double* __restrict__ output,
                              const double* __restrict__ biases,
                              int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;

    double sum = biases[i];
    for (int j = 0; j < cols; ++j) {
        sum += weights[i * cols + j] * input[j];
    }
    output[i] = sum;
}

// ---------------------------------------------------------------------------
// Kernel: ReLU activation  data[i] = max(0, data[i])
// ---------------------------------------------------------------------------
__global__ void kernel_relu(double* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (data[i] < 0.0) data[i] = 0.0;
}

// ---------------------------------------------------------------------------
// Kernel: ReLU derivative element-wise multiply
// result[i] = upstream[i] * (pre_act[i] > 0 ? 1 : 0)
// ---------------------------------------------------------------------------
__global__ void kernel_relu_deriv_mul(const double* __restrict__ pre_act,
                                      const double* __restrict__ upstream,
                                      double* __restrict__ result,
                                      int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    result[i] = (pre_act[i] > 0.0) ? upstream[i] : 0.0;
}

// ---------------------------------------------------------------------------
// Kernel: Softmax (two-pass: find max, then exp-and-normalize)
// Small output size (10), run on a single block with shared memory reduction.
// ---------------------------------------------------------------------------
__global__ void kernel_softmax(double* data, int size) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;

    double val = (tid < size) ? data[tid] : -DBL_MAX;
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction to find max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    double max_val = sdata[0];
    __syncthreads();

    // Compute exp(x - max)
    double exp_val = 0.0;
    if (tid < size) {
        exp_val = exp(data[tid] - max_val);
        sdata[tid] = exp_val;
    } else {
        sdata[tid] = 0.0;
    }
    __syncthreads();

    // Parallel reduction to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    double sum = sdata[0];
    __syncthreads();

    if (tid < size) {
        data[tid] = exp_val / sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel: output delta  delta[i] = output[i] - target[i]
// ---------------------------------------------------------------------------
__global__ void kernel_output_delta(const double* __restrict__ output,
                                    const double* __restrict__ target,
                                    double* __restrict__ delta,
                                    int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    delta[i] = output[i] - target[i];
}

// ---------------------------------------------------------------------------
// Kernel: backprop hidden delta
// delta[i] = relu'(pre_act[i]) * sum_j(weights_next[j*current_size+i] * delta_next[j])
// ---------------------------------------------------------------------------
__global__ void kernel_backprop_delta(const double* __restrict__ weights_next,
                                      const double* __restrict__ delta_next,
                                      const double* __restrict__ pre_act,
                                      double* __restrict__ delta,
                                      int current_size, int next_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= current_size) return;

    double sum = 0.0;
    for (int j = 0; j < next_size; ++j) {
        sum += weights_next[j * current_size + i] * delta_next[j];
    }
    delta[i] = (pre_act[i] > 0.0) ? sum : 0.0;
}

// ---------------------------------------------------------------------------
// Kernel: SGD weight update
// weights[i*cols+j] -= lr * delta[i] * prev_act[j]
// biases[i] -= lr * delta[i]
// ---------------------------------------------------------------------------
__global__ void kernel_update_weights(double* __restrict__ weights,
                                      double* __restrict__ biases,
                                      const double* __restrict__ delta,
                                      const double* __restrict__ prev_act,
                                      int rows, int cols, double lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int i = idx / cols;
        int j = idx % cols;
        weights[idx] -= lr * delta[i] * prev_act[j];
    }

    // First `rows` threads also update biases
    if (idx < rows) {
        biases[idx] -= lr * delta[idx];
    }
}


// ===========================================================================
// GpuBackend method implementations
// ===========================================================================

bool KernelLog::enabled = true;
bool OpLog::cpu_enabled = false;
bool OpLog::gpu_enabled = false;

GpuBackend& GpuBackend::instance() {
    static GpuBackend g;
    return g;
}

GpuBackend::~GpuBackend() {
    shutdown();
}

bool GpuBackend::initialize() {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        std::cout << "[GPU] No AMD GPU found. Falling back to CPU." << std::endl;
        available_ = false;
        return false;
    }

    hipDeviceProp_t props;
    (void)hipGetDeviceProperties(&props, 0);
    device_name_ = props.name;

    std::cout << "[GPU] AMD GPU detected: " << device_name_ << std::endl;
    std::cout << "[GPU] Compute units: " << props.multiProcessorCount << std::endl;
    std::cout << "[GPU] Global memory: " << (props.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "[GPU] Max threads/block: " << props.maxThreadsPerBlock << std::endl;

    (void)hipSetDevice(0);
    available_ = true;
    return true;
}

void GpuBackend::shutdown() {
    if (available_) {
        // Do NOT call hipDeviceReset() here. It invalidates all device
        // allocations and corrupts the glibc heap when HIP's internal
        // bookkeeping interacts with malloc during static destruction.
        // The OS reclaims all GPU resources on process exit.
        available_ = false;
    }
}

void GpuBackend::set_kernel_logging(bool enabled) {
    KernelLog::enabled = enabled;
}

double* GpuBackend::alloc(size_t count) {
    double* ptr = nullptr;
    size_t bytes = count * sizeof(double);
    (void)hipMalloc(&ptr, bytes);
    OpLog::gpu_mem("hipMalloc (GPU memory allocated)", bytes, ptr);
    return ptr;
}

void GpuBackend::free(double* ptr) {
    if (ptr) {
        OpLog::gpu_mem("hipFree (GPU memory freed)", 0, ptr);
        (void)hipFree(ptr);
    }
}

void GpuBackend::copy_to_device(double* d_dst, const double* h_src, size_t count) {
    size_t bytes = count * sizeof(double);
    OpLog::transfer("Copy HOST -> DEVICE (CPU to GPU)", bytes);
    (void)hipMemcpy(d_dst, h_src, bytes, hipMemcpyHostToDevice);
}

void GpuBackend::copy_to_host(double* h_dst, const double* d_src, size_t count) {
    size_t bytes = count * sizeof(double);
    OpLog::transfer("Copy DEVICE -> HOST (GPU to CPU)", bytes);
    (void)hipMemcpy(h_dst, d_src, bytes, hipMemcpyDeviceToHost);
}

void GpuBackend::copy_on_device(double* d_dst, const double* d_src, size_t count) {
    size_t bytes = count * sizeof(double);
    OpLog::transfer("Copy DEVICE -> DEVICE (GPU to GPU)", bytes);
    (void)hipMemcpy(d_dst, d_src, bytes, hipMemcpyDeviceToDevice);
}

void GpuBackend::synchronize() {
    OpLog::gpu_phase("hipDeviceSynchronize -- waiting for all GPU kernels to finish...");
    (void)hipDeviceSynchronize();
    OpLog::gpu_phase("hipDeviceSynchronize -- complete");
}

void GpuBackend::matvec_multiply(const double* d_weights, const double* d_input,
                                  double* d_output, const double* d_biases,
                                  int rows, int cols) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    KernelLog::log("kernel_matvec", grid, block);
    hipLaunchKernelGGL(kernel_matvec, grid, block, 0, 0,
                       d_weights, d_input, d_output, d_biases, rows, cols);
    KernelLog::done("kernel_matvec");
}

void GpuBackend::relu_forward(double* d_data, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    KernelLog::log("kernel_relu", grid, block);
    hipLaunchKernelGGL(kernel_relu, grid, block, 0, 0, d_data, size);
    KernelLog::done("kernel_relu");
}

void GpuBackend::relu_derivative_multiply(const double* d_pre_act, const double* d_upstream,
                                           double* d_result, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    KernelLog::log("kernel_relu_deriv_mul", grid, block);
    hipLaunchKernelGGL(kernel_relu_deriv_mul, grid, block, 0, 0,
                       d_pre_act, d_upstream, d_result, size);
    KernelLog::done("kernel_relu_deriv_mul");
}

void GpuBackend::softmax_forward(double* d_data, int size) {
    int threads = 1;
    while (threads < size) threads *= 2;
    if (threads < 32) threads = 32;
    size_t shared = threads * sizeof(double);
    dim3 grid(1), block(threads);

    KernelLog::log("kernel_softmax", grid, block, shared);
    hipLaunchKernelGGL(kernel_softmax, grid, block, shared, 0, d_data, size);
    KernelLog::done("kernel_softmax");
}

void GpuBackend::compute_output_delta(const double* d_output, const double* d_target,
                                       double* d_delta, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    KernelLog::log("kernel_output_delta", grid, block);
    hipLaunchKernelGGL(kernel_output_delta, grid, block, 0, 0,
                       d_output, d_target, d_delta, size);
    KernelLog::done("kernel_output_delta");
}

void GpuBackend::backprop_delta(const double* d_weights_next, const double* d_delta_next,
                                 const double* d_pre_act, double* d_delta,
                                 int current_size, int next_size) {
    int threads = 256;
    int blocks = (current_size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    KernelLog::log("kernel_backprop_delta", grid, block);
    hipLaunchKernelGGL(kernel_backprop_delta, grid, block, 0, 0,
                       d_weights_next, d_delta_next, d_pre_act, d_delta,
                       current_size, next_size);
    KernelLog::done("kernel_backprop_delta");
}

void GpuBackend::update_weights(double* d_weights, double* d_biases,
                                 const double* d_delta, const double* d_prev_act,
                                 int rows, int cols, double learning_rate) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    KernelLog::log("kernel_update_weights", grid, block);
    hipLaunchKernelGGL(kernel_update_weights, grid, block, 0, 0,
                       d_weights, d_biases, d_delta, d_prev_act,
                       rows, cols, learning_rate);
    KernelLog::done("kernel_update_weights");
}

} // namespace digitrec
