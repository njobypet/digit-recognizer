#include <hip/hip_runtime.h>
#include "gpu_backend.h"
#include <cfloat>
#include <cmath>
#include <thread>
#include <random>

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


// ---------------------------------------------------------------------------
// Kernel: flip every bit in a buffer (used by --gpumem)
// ---------------------------------------------------------------------------
__global__ void kernel_bitflip(unsigned char* data, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    data[i] = ~data[i];
}

// ---------------------------------------------------------------------------
// Kernel: zero out a buffer (used by --gpumem)
// ---------------------------------------------------------------------------
__global__ void kernel_memzero(unsigned char* data, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    data[i] = 0;
}

// ===========================================================================
// GpuBackend method implementations
// ===========================================================================

bool KernelLog::enabled = false;
bool OpLog::cpu_enabled = false;
bool OpLog::gpu_enabled = false;
bool GpuDelay::enabled = false;

static std::mt19937& delay_rng() {
    static std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    return rng;
}

bool GpuDelay::should_delay() {
    if (!enabled) return false;
    std::uniform_int_distribution<int> dist(1, 100);
    return dist(delay_rng()) <= probability_percent;
}

int GpuDelay::random_delay_ms() {
    std::uniform_int_distribution<int> dist(min_delay_ms, max_delay_ms);
    return dist(delay_rng());
}

void GpuDelay::apply(const char* kernel_name, std::string& display_name) {
    if (!should_delay()) {
        display_name = kernel_name;
        return;
    }
    display_name = std::string(kernel_name) + "_delay";
    int delay = random_delay_ms();
    std::cout << "[DELAY " << OpLog::now_ms() << "ms] Injecting " << delay
              << "ms delay before " << display_name << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
}

bool GpuMemSpike::enabled = false;

bool GpuMemSpike::should_spike() {
    if (!enabled) return false;
    std::uniform_int_distribution<int> dist(1, 100);
    return dist(delay_rng()) <= probability_percent;
}

size_t GpuMemSpike::random_size() {
    std::uniform_int_distribution<size_t> dist(min_bytes, max_bytes);
    return dist(delay_rng());
}

void GpuMemSpike::apply(const char* kernel_name, std::string& display_name,
                         const void* d_image_src, size_t image_bytes) {
    if (!should_spike()) return;

    size_t bytes = random_size();
    double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);

    display_name = std::string(kernel_name) + "_mem";

    unsigned char* ptr = nullptr;
    (void)hipMalloc(&ptr, bytes);

    std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Allocated "
              << static_cast<int>(mb) << " MB (" << bytes << " bytes)"
              << " for " << display_name
              << "  ptr=" << static_cast<void*>(ptr) << std::endl;

    // Step 1: Fill allocation with image data (tiled to fill the entire buffer)
    if (d_image_src && image_bytes > 0 && ptr) {
        size_t filled = 0;
        while (filled < bytes) {
            size_t chunk = std::min(image_bytes, bytes - filled);
            (void)hipMemcpy(ptr + filled, d_image_src, chunk, hipMemcpyDeviceToDevice);
            filled += chunk;
        }
        (void)hipDeviceSynchronize();
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Filled with image data (tiled "
                  << image_bytes << " bytes across " << bytes << " bytes)"
                  << " for " << display_name << std::endl;
    }

    // Step 2: Copy GPU memory to CPU, flip every bit on CPU, copy back to GPU
    if (ptr) {
        unsigned char* h_buf = new unsigned char[bytes];

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] hipMemcpy GPU -> CPU ("
                  << bytes << " bytes) for " << display_name << std::endl;
        (void)hipMemcpy(h_buf, ptr, bytes, hipMemcpyDeviceToHost);

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Flipping all bits on CPU ("
                  << bytes << " bytes) for " << display_name << std::endl;
        for (size_t i = 0; i < bytes; ++i) {
            h_buf[i] = ~h_buf[i];
        }

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] hipMemcpy CPU -> GPU ("
                  << bytes << " bytes) for " << display_name << std::endl;
        (void)hipMemcpy(ptr, h_buf, bytes, hipMemcpyHostToDevice);

        delete[] h_buf;
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Bitflip round-trip complete"
                  << " for " << display_name << std::endl;
    }

    // Step 3: Zero out the GPU memory via kernel
    if (ptr) {
        int threads = 256;
        size_t blocks = (bytes + threads - 1) / threads;
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Launching kernel_memzero ("
                  << bytes << " bytes) for " << display_name << std::endl;
        hipLaunchKernelGGL(kernel_memzero, dim3(blocks), dim3(threads), 0, 0, ptr, bytes);
        (void)hipDeviceSynchronize();
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] kernel_memzero complete"
                  << " for " << display_name << std::endl;
    }

    // Step 4: Free the allocation
    (void)hipFree(ptr);

    std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Freed "
              << static_cast<int>(mb) << " MB"
              << " for " << display_name << std::endl;
}

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

    std::string name;
    GpuDelay::apply("kernel_matvec", name);
    GpuMemSpike::apply("kernel_matvec", name, d_input, cols * sizeof(double));
    KernelLog::log(name.c_str(), grid, block);
    hipLaunchKernelGGL(kernel_matvec, grid, block, 0, 0,
                       d_weights, d_input, d_output, d_biases, rows, cols);
    KernelLog::done(name.c_str());
}

void GpuBackend::relu_forward(double* d_data, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    std::string name;
    GpuDelay::apply("kernel_relu", name);
    GpuMemSpike::apply("kernel_relu", name, d_data, size * sizeof(double));
    KernelLog::log(name.c_str(), grid, block);
    hipLaunchKernelGGL(kernel_relu, grid, block, 0, 0, d_data, size);
    KernelLog::done(name.c_str());
}

void GpuBackend::relu_derivative_multiply(const double* d_pre_act, const double* d_upstream,
                                           double* d_result, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    std::string name;
    GpuDelay::apply("kernel_relu_deriv_mul", name);
    GpuMemSpike::apply("kernel_relu_deriv_mul", name, d_pre_act, size * sizeof(double));
    KernelLog::log(name.c_str(), grid, block);
    hipLaunchKernelGGL(kernel_relu_deriv_mul, grid, block, 0, 0,
                       d_pre_act, d_upstream, d_result, size);
    KernelLog::done(name.c_str());
}

void GpuBackend::softmax_forward(double* d_data, int size) {
    int threads = 1;
    while (threads < size) threads *= 2;
    if (threads < 32) threads = 32;
    size_t shared = threads * sizeof(double);
    dim3 grid(1), block(threads);

    std::string name;
    GpuDelay::apply("kernel_softmax", name);
    GpuMemSpike::apply("kernel_softmax", name, d_data, size * sizeof(double));
    KernelLog::log(name.c_str(), grid, block, shared);
    hipLaunchKernelGGL(kernel_softmax, grid, block, shared, 0, d_data, size);
    KernelLog::done(name.c_str());
}

void GpuBackend::compute_output_delta(const double* d_output, const double* d_target,
                                       double* d_delta, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    std::string name;
    GpuDelay::apply("kernel_output_delta", name);
    GpuMemSpike::apply("kernel_output_delta", name, d_output, size * sizeof(double));
    KernelLog::log(name.c_str(), grid, block);
    hipLaunchKernelGGL(kernel_output_delta, grid, block, 0, 0,
                       d_output, d_target, d_delta, size);
    KernelLog::done(name.c_str());
}

void GpuBackend::backprop_delta(const double* d_weights_next, const double* d_delta_next,
                                 const double* d_pre_act, double* d_delta,
                                 int current_size, int next_size) {
    int threads = 256;
    int blocks = (current_size + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    std::string name;
    GpuDelay::apply("kernel_backprop_delta", name);
    GpuMemSpike::apply("kernel_backprop_delta", name, d_pre_act, current_size * sizeof(double));
    KernelLog::log(name.c_str(), grid, block);
    hipLaunchKernelGGL(kernel_backprop_delta, grid, block, 0, 0,
                       d_weights_next, d_delta_next, d_pre_act, d_delta,
                       current_size, next_size);
    KernelLog::done(name.c_str());
}

void GpuBackend::update_weights(double* d_weights, double* d_biases,
                                 const double* d_delta, const double* d_prev_act,
                                 int rows, int cols, double learning_rate) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dim3 grid(blocks), block(threads);

    std::string name;
    GpuDelay::apply("kernel_update_weights", name);
    GpuMemSpike::apply("kernel_update_weights", name, d_prev_act, cols * sizeof(double));
    KernelLog::log(name.c_str(), grid, block);
    hipLaunchKernelGGL(kernel_update_weights, grid, block, 0, 0,
                       d_weights, d_biases, d_delta, d_prev_act,
                       rows, cols, learning_rate);
    KernelLog::done(name.c_str());
}

} // namespace digitrec
