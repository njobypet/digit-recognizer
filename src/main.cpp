#include "digit_recognizer.h"
#include "gpu_backend.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>

#ifndef USE_HIP
bool digitrec::OpLog::enabled = false;
#endif

void print_usage(const char* program) {
    std::cout << "Digit Recognizer - Handwritten Digit OCR (Numbers Only)\n"
              << "========================================================\n\n"
              << "Usage:\n"
              << "  " << program << " train <mnist_dir> [options]\n"
              << "  " << program << " test <mnist_dir> --model <model_file>\n"
              << "  " << program << " predict <image_file> --model <model_file>\n"
              << "  " << program << " predict-multi <image_file> --model <model_file>\n\n"
              << "Commands:\n"
              << "  train          Train the model on MNIST dataset\n"
              << "  test           Evaluate model accuracy on MNIST test set\n"
              << "  predict        Recognize a single digit from an image\n"
              << "  predict-multi  Recognize multiple digits from an image\n\n"
              << "Options:\n"
              << "  --model <file>   Path to model file (default: digit_model.bin)\n"
              << "  --epochs <n>     Training epochs (default: 10)\n"
              << "  --batch <n>      Batch size (default: 32)\n"
              << "  --lr <rate>      Learning rate (default: 0.005)\n"
              << "  --gpu            Use AMD GPU acceleration (ROCm/HIP)\n"
              << "  --verbose        Print detailed logs (memory, kernels, data transfers)\n\n"
              << "Examples:\n"
              << "  " << program << " train ./data --epochs 15 --gpu --model my_model.bin\n"
              << "  " << program << " predict digit.png --model my_model.bin --verbose\n"
              << "  " << program << " predict digit.png --model my_model.bin --gpu --verbose\n";
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
};

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

void enable_verbose(const Config& config) {
    if (!config.verbose) return;
    digitrec::OpLog::enabled = true;
    auto& gpu = digitrec::GpuBackend::instance();
    gpu.set_verbose(true);
}

int cmd_train(const Config& config) {
    digitrec::DigitRecognizer recognizer;
    try_init_gpu(recognizer, config.use_gpu);

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
    recognizer.test_on_mnist(config.input_path);
    return 0;
}

int cmd_predict(const Config& config) {
    digitrec::DigitRecognizer recognizer;

    if (!recognizer.load_model(config.model_path)) {
        std::cerr << "Error: Could not load model from " << config.model_path << std::endl;
        return 1;
    }

    try_init_gpu(recognizer, config.use_gpu);
    enable_verbose(config);

    try {
        auto result = recognizer.recognize(config.input_path);

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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int cmd_predict_multi(const Config& config) {
    digitrec::DigitRecognizer recognizer;

    if (!recognizer.load_model(config.model_path)) {
        std::cerr << "Error: Could not load model from " << config.model_path << std::endl;
        return 1;
    }

    try_init_gpu(recognizer, config.use_gpu);
    enable_verbose(config);

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
