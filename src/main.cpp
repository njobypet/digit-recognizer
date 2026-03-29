#include "digit_recognizer.h"
#include "gpu_backend.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>
#include <csignal>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

#ifndef USE_HIP
bool digitrec::OpLog::cpu_enabled = false;
bool digitrec::OpLog::gpu_enabled = false;
bool digitrec::GpuDelay::enabled = false;

bool digitrec::GpuDelay::should_delay() { return false; }
int digitrec::GpuDelay::random_delay_ms() { return 0; }
void digitrec::GpuDelay::apply(const char* kernel_name, std::string& display_name) {
    display_name = kernel_name;
}
#endif

static const char* PID_FILE = ".digit_recognizer.pid";
static const char* STOP_FILE = ".digit_recognizer.stop";

static volatile bool g_running = true;

static void signal_handler(int) {
    g_running = false;
}

void print_usage(const char* program) {
    std::cout << "Digit Recognizer - Handwritten Digit OCR (Numbers Only)\n"
              << "========================================================\n\n"
              << "Usage:\n"
              << "  " << program << " train <mnist_dir> [options]\n"
              << "  " << program << " test <mnist_dir> --model <model_file>\n"
              << "  " << program << " predict <image_file_or_dir> --model <model_file>\n"
              << "  " << program << " predict-multi <image_file> --model <model_file>\n\n"
              << "Commands:\n"
              << "  train          Train the model on MNIST dataset\n"
              << "  test           Evaluate model accuracy on MNIST test set\n"
              << "  predict        Recognize a single digit from an image\n"
              << "  predict-multi  Recognize multiple digits from an image\n\n"
              << "Options:\n"
              << "  --model <file>     Path to model file (default: digit_model.bin)\n"
              << "  --epochs <n>       Training epochs (default: 10)\n"
              << "  --batch <n>        Batch size (default: 32)\n"
              << "  --lr <rate>        Learning rate (default: 0.005)\n"
              << "  --gpu              Use AMD GPU acceleration (ROCm/HIP)\n"
              << "  --verbose          Enable all logs (CPU + GPU)\n"
              << "  --cpulogs on|off   Turn CPU logs on or off\n"
              << "  --gpulogs on|off   Turn GPU logs on or off\n"
              << "  --infinite         Run predict in an infinite loop on random images\n"
              << "                     (pass a directory as <image_file_or_dir>)\n"
              << "                     Stop with: stop_digit_recognizer or Ctrl+C\n"
              << "  --gpudelay         Inject random 1-100ms delays into ~10% of GPU kernels\n"
              << "                     Delayed kernels are suffixed '_delay' in logs\n\n"
              << "Examples:\n"
              << "  " << program << " predict digit.png --model m.bin\n"
              << "  " << program << " predict sample_images --model m.bin --infinite --gpu\n"
              << "  " << program << " predict sample_images --model m.bin --infinite --gpulogs on\n";
}

struct Config {
    std::string command;
    std::string input_path;
    std::string model_path = "digit_model.bin";
    int epochs = 10;
    int batch_size = 32;
    double learning_rate = 0.005;
    bool use_gpu = false;
    bool verbose = false;
    bool infinite = false;
    bool gpudelay = false;
    int cpulogs = -1;
    int gpulogs = -1;
};

bool parse_on_off(const char* value) {
    std::string v = value;
    return (v == "on" || v == "1" || v == "true" || v == "yes");
}

bool parse_args(int argc, char* argv[], Config& config) {
    if (argc < 3) return false;

    config.command = argv[1];
    config.input_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stod(argv[++i]);
        } else if (arg == "--gpu") {
            config.use_gpu = true;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--infinite") {
            config.infinite = true;
        } else if (arg == "--gpudelay") {
            config.gpudelay = true;
        } else if (arg == "--cpulogs" && i + 1 < argc) {
            config.cpulogs = parse_on_off(argv[++i]) ? 1 : 0;
        } else if (arg == "--gpulogs" && i + 1 < argc) {
            config.gpulogs = parse_on_off(argv[++i]) ? 1 : 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

void try_init_gpu(digitrec::DigitRecognizer& recognizer, bool requested) {
    if (!requested) return;
    if (!recognizer.init_gpu()) {
        std::cerr << "[GPU] Warning: GPU initialization failed. Continuing on CPU." << std::endl;
    }
}

void apply_log_flags(const Config& config) {
    bool cpu_on = config.verbose;
    bool gpu_on = config.verbose;

    if (config.cpulogs != -1) cpu_on = (config.cpulogs == 1);
    if (config.gpulogs != -1) gpu_on = (config.gpulogs == 1);

    digitrec::OpLog::cpu_enabled = cpu_on;
    digitrec::OpLog::gpu_enabled = gpu_on;
    digitrec::GpuDelay::enabled = config.gpudelay;

#ifdef USE_HIP
    digitrec::KernelLog::enabled = gpu_on;
#endif
}

void write_pid_file() {
#ifdef _WIN32
    int pid = static_cast<int>(_getpid());
#else
    int pid = static_cast<int>(getpid());
#endif
    std::ofstream f(PID_FILE);
    f << pid << std::endl;
}

void cleanup_signal_files() {
    std::error_code ec;
    fs::remove(PID_FILE, ec);
    fs::remove(STOP_FILE, ec);
}

bool stop_requested() {
    return fs::exists(STOP_FILE);
}

std::vector<std::string> collect_images(const std::string& dir) {
    std::vector<std::string> files;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
            files.push_back(entry.path().string());
        }
    }
    return files;
}

int cmd_train(const Config& config) {
    digitrec::DigitRecognizer recognizer;
    try_init_gpu(recognizer, config.use_gpu);
    apply_log_flags(config);

    recognizer.train_on_mnist(config.input_path, config.epochs, config.batch_size);
    recognizer.save_model(config.model_path);

    return 0;
}

int cmd_test(const Config& config) {
    digitrec::DigitRecognizer recognizer;

    if (!recognizer.load_model(config.model_path)) {
        std::cerr << "Error: Could not load model from " << config.model_path << std::endl;
        return 1;
    }

    try_init_gpu(recognizer, config.use_gpu);
    apply_log_flags(config);
    recognizer.test_on_mnist(config.input_path);
    return 0;
}

void print_prediction(const digitrec::PredictionResult& result) {
    std::cout << "\nPrediction Results\n"
              << std::string(30, '-') << "\n"
              << "Predicted digit: " << result.digit << "\n"
              << "Confidence:      " << std::fixed << std::setprecision(1)
              << (result.confidence * 100.0) << "%\n\n"
              << "All probabilities:\n";

    for (int i = 0; i < 10; ++i) {
        std::cout << "  " << i << ": "
                  << std::setprecision(2) << (result.probabilities[i] * 100.0) << "%";
        if (i == result.digit) std::cout << "  <-- predicted";
        std::cout << "\n";
    }
}

int cmd_predict(const Config& config) {
    digitrec::DigitRecognizer recognizer;

    if (!recognizer.load_model(config.model_path)) {
        std::cerr << "Error: Could not load model from " << config.model_path << std::endl;
        return 1;
    }

    try_init_gpu(recognizer, config.use_gpu);
    apply_log_flags(config);

    if (!config.infinite) {
        try {
            auto result = recognizer.recognize(config.input_path);
            print_prediction(result);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }

    // --- Infinite loop mode ---
    auto image_files = collect_images(config.input_path);
    if (image_files.empty()) {
        std::cerr << "No image files found in " << config.input_path << std::endl;
        return 1;
    }

    signal(SIGINT, signal_handler);
#ifdef SIGTERM
    signal(SIGTERM, signal_handler);
#endif

    cleanup_signal_files();
    write_pid_file();

    std::cout << "=== Infinite predict mode ===" << std::endl;
    std::cout << "Images dir:  " << config.input_path << std::endl;
    std::cout << "Image count: " << image_files.size() << std::endl;
    std::cout << "GPU:         " << (config.use_gpu ? "yes" : "no") << std::endl;
    std::cout << "PID file:    " << PID_FILE << std::endl;
    std::cout << "Stop with:   stop_digit_recognizer  or  Ctrl+C\n" << std::endl;

    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<size_t> dist(0, image_files.size() - 1);

    int count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (g_running && !stop_requested()) {
        size_t idx = dist(rng);
        const auto& img_path = image_files[idx];
        count++;

        try {
            auto result = recognizer.recognize(img_path);
            std::cout << "[" << std::setw(6) << count << "] "
                      << "digit=" << result.digit
                      << "  conf=" << std::fixed << std::setprecision(1)
                      << (result.confidence * 100.0) << "%"
                      << "  file=" << fs::path(img_path).filename().string()
                      << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[" << std::setw(6) << count << "] "
                      << "ERROR: " << e.what()
                      << "  file=" << fs::path(img_path).filename().string()
                      << std::endl;
        }
    }

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    double secs = std::chrono::duration<double>(elapsed).count();

    std::cout << "\n=== Stopped ===" << std::endl;
    if (stop_requested()) {
        std::cout << "Reason: stop_digit_recognizer signal received" << std::endl;
    } else {
        std::cout << "Reason: Ctrl+C / SIGINT" << std::endl;
    }
    std::cout << "Total predictions: " << count << std::endl;
    std::cout << "Elapsed: " << std::fixed << std::setprecision(1) << secs << "s" << std::endl;
    if (count > 0) {
        std::cout << "Avg per prediction: " << std::setprecision(0)
                  << (secs / count * 1000.0) << "ms" << std::endl;
    }

    cleanup_signal_files();
    return 0;
}

int cmd_predict_multi(const Config& config) {
    digitrec::DigitRecognizer recognizer;

    if (!recognizer.load_model(config.model_path)) {
        std::cerr << "Error: Could not load model from " << config.model_path << std::endl;
        return 1;
    }

    try_init_gpu(recognizer, config.use_gpu);
    apply_log_flags(config);

    try {
        auto result = recognizer.recognize_multi_digit(config.input_path);
        std::cout << "\nRecognized number: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    Config config;

    if (!parse_args(argc, argv, config)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        if (config.command == "train") {
            return cmd_train(config);
        } else if (config.command == "test") {
            return cmd_test(config);
        } else if (config.command == "predict") {
            return cmd_predict(config);
        } else if (config.command == "predict-multi") {
            return cmd_predict_multi(config);
        } else {
            std::cerr << "Unknown command: " << config.command << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
