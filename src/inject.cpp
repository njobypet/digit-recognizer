#include "gpu_backend.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace digitrec {

bool OpLog::cpu_enabled = false;
bool OpLog::gpu_enabled = false;
bool GpuDelay::enabled = false;
bool GpuMemSpike::enabled = false;

static std::mt19937& inject_rng() {
    static std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    return rng;
}

bool GpuDelay::should_delay() {
    if (!enabled) return false;
    std::uniform_int_distribution<int> dist(1, 100);
    return dist(inject_rng()) <= probability_percent;
}

int GpuDelay::random_delay_ms() {
    std::uniform_int_distribution<int> dist(min_delay_ms, max_delay_ms);
    return dist(inject_rng());
}

void GpuDelay::apply(const char* kernel_name, std::string& display_name) {
    if (display_name.empty()) {
        display_name = kernel_name;
    }
    if (!should_delay()) {
        return;
    }
    display_name += "_delay";
    int delay = random_delay_ms();
    std::cout << "[DELAY " << OpLog::now_ms() << "ms] Injecting " << delay
              << "ms delay before " << display_name << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
}

bool GpuMemSpike::should_spike() {
    if (!enabled) return false;
    std::uniform_int_distribution<int> dist(1, 100);
    return dist(inject_rng()) <= probability_percent;
}

size_t GpuMemSpike::random_size() {
    std::uniform_int_distribution<unsigned long long> dist(
        static_cast<unsigned long long>(min_bytes),
        static_cast<unsigned long long>(max_bytes));
    return static_cast<size_t>(dist(inject_rng()));
}

void GpuMemSpike::apply(const char* kernel_name, std::string& display_name,
                        const void* image_src, size_t image_bytes,
                        bool src_on_device) {
    if (display_name.empty()) {
        display_name = kernel_name;
    }
    if (!should_spike()) {
        return;
    }

    display_name += "_mem";
    size_t bytes = random_size();
    double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);

#ifdef USE_HIP
    unsigned char* ptr = nullptr;
    (void)hipMalloc(&ptr, bytes);

    std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Allocated "
              << static_cast<int>(mb) << " MB (" << bytes << " bytes)"
              << " for " << display_name
              << "  ptr=" << static_cast<void*>(ptr) << std::endl;

    if (ptr && image_src && image_bytes > 0) {
        size_t filled = 0;
        while (filled < bytes) {
            size_t chunk = std::min(image_bytes, bytes - filled);
            hipMemcpyKind kind = src_on_device ? hipMemcpyDeviceToDevice
                                               : hipMemcpyHostToDevice;
            (void)hipMemcpy(ptr + filled, image_src, chunk, kind);
            filled += chunk;
        }
        (void)hipDeviceSynchronize();
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Filled with image data (tiled "
                  << image_bytes << " bytes across " << bytes << " bytes)"
                  << " for " << display_name << std::endl;
    }

    if (ptr) {
        std::vector<unsigned char> h_buf(bytes);

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] hipMemcpy GPU -> CPU ("
                  << bytes << " bytes) for " << display_name << std::endl;
        (void)hipMemcpy(h_buf.data(), ptr, bytes, hipMemcpyDeviceToHost);

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Flipping all bits on CPU ("
                  << bytes << " bytes) for " << display_name << std::endl;
        for (size_t i = 0; i < bytes; ++i) {
            h_buf[i] = static_cast<unsigned char>(~h_buf[i]);
        }

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] hipMemcpy CPU -> GPU ("
                  << bytes << " bytes) for " << display_name << std::endl;
        (void)hipMemcpy(ptr, h_buf.data(), bytes, hipMemcpyHostToDevice);

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Bitflip round-trip complete"
                  << " for " << display_name << std::endl;

        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Launching kernel_memzero ("
                  << bytes << " bytes) for " << display_name << std::endl;
        launch_kernel_memzero(ptr, bytes);
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] kernel_memzero complete"
                  << " for " << display_name << std::endl;

        (void)hipFree(ptr);
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Freed "
                  << static_cast<int>(mb) << " MB"
                  << " for " << display_name << std::endl;
    }
#else
    (void)src_on_device;
    std::vector<unsigned char> buf(bytes);

    std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Allocated "
              << static_cast<int>(mb) << " MB (" << bytes << " bytes)"
              << " for " << display_name
              << "  ptr=" << static_cast<void*>(buf.data()) << std::endl;

    if (image_src && image_bytes > 0) {
        size_t filled = 0;
        const auto* src = static_cast<const unsigned char*>(image_src);
        while (filled < bytes) {
            size_t chunk = std::min(image_bytes, bytes - filled);
            std::copy(src, src + chunk, buf.data() + filled);
            filled += chunk;
        }
        std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Filled with image data (tiled "
                  << image_bytes << " bytes across " << bytes << " bytes)"
                  << " for " << display_name << std::endl;
    }

    std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Flipping all bits on CPU ("
              << bytes << " bytes) for " << display_name << std::endl;
    for (size_t i = 0; i < bytes; ++i) {
        buf[i] = static_cast<unsigned char>(~buf[i]);
    }

    std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Zeroing buffer ("
              << bytes << " bytes) for " << display_name << std::endl;
    std::fill(buf.begin(), buf.end(), static_cast<unsigned char>(0));

    std::cout << "[GPUMEM " << OpLog::now_ms() << "ms] Freed "
              << static_cast<int>(mb) << " MB"
              << " for " << display_name << std::endl;
#endif
}

} // namespace digitrec
