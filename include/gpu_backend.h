#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cstdint>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace digitrec {

struct OpLog {
    static bool cpu_enabled;
    static bool gpu_enabled;

    static long long now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static void cpu(const char* op, size_t bytes = 0) {
        if (!cpu_enabled) return;
        std::cout << "[CPU " << now_ms() << "ms] " << op;
        if (bytes > 0) std::cout << "  size=" << bytes << " bytes";
        std::cout << std::endl;
    }

    static void gpu_mem(const char* op, size_t bytes, const void* ptr = nullptr) {
        if (!gpu_enabled) return;
        std::cout << "[GPU " << now_ms() << "ms] " << op;
        if (bytes > 0) std::cout << "  size=" << bytes << " bytes";
        if (ptr) std::cout << "  ptr=" << ptr;
        std::cout << std::endl;
    }

    static void transfer(const char* direction, size_t bytes) {
        if (!gpu_enabled) return;
        std::cout << "[MEM " << now_ms() << "ms] " << direction
                  << "  size=" << bytes << " bytes" << std::endl;
    }

    static void gpu_phase(const char* msg) {
        if (!gpu_enabled) return;
        std::cout << "[GPU " << now_ms() << "ms] " << msg << std::endl;
    }

    static void cpu_phase(const char* msg) {
        if (!cpu_enabled) return;
        std::cout << "[CPU " << now_ms() << "ms] " << msg << std::endl;
    }

    static void phase(const char* msg) {
        if (!cpu_enabled && !gpu_enabled) return;
        std::cout << "[LOG " << now_ms() << "ms] " << msg << std::endl;
    }
};

struct GpuDelay {
    static bool enabled;
    static constexpr int probability_percent = 10;
    static constexpr int min_delay_ms = 1;
    static constexpr int max_delay_ms = 100;

    static bool should_delay();
    static int random_delay_ms();
    static void apply(const char* kernel_name, std::string& display_name);
};

struct GpuMemSpike {
    static bool enabled;
    static constexpr int probability_percent = 10;
    static constexpr size_t min_bytes = 1 * 1024 * 1024;      // 1 MB
    static constexpr size_t max_bytes = 100 * 1024 * 1024;    // 100 MB

    static bool should_spike();
    static size_t random_size();
    static void apply(const char* kernel_name, std::string& display_name);
};

#ifdef USE_HIP

struct KernelLog {
    static bool enabled;

    static void log(const char* kernel_name, dim3 grid, dim3 block, size_t shared_mem = 0) {
        if (!enabled && !OpLog::gpu_enabled) return;
        auto ms = OpLog::now_ms();
        std::cout << "[GPU " << ms << "ms] Launching kernel: " << kernel_name
                  << "  grid(" << grid.x << "," << grid.y << "," << grid.z << ")"
                  << "  block(" << block.x << "," << block.y << "," << block.z << ")"
                  << (shared_mem > 0 ? "  shared=" + std::to_string(shared_mem) + "B" : "")
                  << std::endl;
    }

    static void done(const char* kernel_name) {
        if (!enabled && !OpLog::gpu_enabled) return;
        std::cout << "[GPU " << OpLog::now_ms() << "ms] Kernel complete: "
                  << kernel_name << std::endl;
    }
};

class GpuBackend {
public:
    static GpuBackend& instance();

    bool initialize();
    void shutdown();
    bool is_available() const { return available_; }
    std::string device_name() const { return device_name_; }

    void matvec_multiply(const double* d_weights, const double* d_input, double* d_output,
                         const double* d_biases, int rows, int cols);

    void relu_forward(double* d_data, int size);

    void relu_derivative_multiply(const double* d_pre_act, const double* d_upstream,
                                  double* d_result, int size);

    void softmax_forward(double* d_data, int size);

    void compute_output_delta(const double* d_output, const double* d_target,
                              double* d_delta, int size);

    void backprop_delta(const double* d_weights_next, const double* d_delta_next,
                        const double* d_pre_act, double* d_delta,
                        int current_size, int next_size);

    void update_weights(double* d_weights, double* d_biases,
                        const double* d_delta, const double* d_prev_act,
                        int rows, int cols, double learning_rate);

    double* alloc(size_t count);
    void free(double* ptr);
    void copy_to_device(double* d_dst, const double* h_src, size_t count);
    void copy_to_host(double* h_dst, const double* d_src, size_t count);
    void copy_on_device(double* d_dst, const double* d_src, size_t count);
    void synchronize();

    void set_kernel_logging(bool enabled);

private:
    GpuBackend() = default;
    ~GpuBackend();

    bool available_ = false;
    std::string device_name_;
};

#else // !USE_HIP

class GpuBackend {
public:
    static GpuBackend& instance() { static GpuBackend g; return g; }
    bool initialize() {
        std::cout << "[GPU] ROCm/HIP not available (built without USE_HIP). Using CPU." << std::endl;
        return false;
    }
    void shutdown() {}
    bool is_available() const { return false; }
    std::string device_name() const { return "CPU (no GPU)"; }
    void set_kernel_logging(bool) {}
};

#endif // USE_HIP

} // namespace digitrec
