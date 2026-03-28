#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cstdlib>
#include <csignal>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

static volatile bool running = true;

static void signal_handler(int) {
    running = false;
}

struct Stats {
    int total = 0;
    int correct = 0;
    int wrong = 0;
    int errors = 0;
};

int extract_expected_digit(const std::string& filename) {
    // Filenames like "digit_7_3.bmp" -> extract 7
    auto base = fs::path(filename).stem().string();
    if (base.rfind("digit_", 0) == 0 && base.size() >= 8) {
        return base[6] - '0';
    }
    return -1;
}

int parse_predicted_digit(const std::string& output) {
    // Look for "Predicted digit: X" in the output
    auto pos = output.find("Predicted digit: ");
    if (pos != std::string::npos) {
        pos += 17;
        if (pos < output.size() && output[pos] >= '0' && output[pos] <= '9') {
            return output[pos] - '0';
        }
    }
    return -1;
}

std::string run_command(const std::string& cmd, int& exit_code) {
    std::string result;
    std::string full_cmd = cmd + " 2>&1";

#ifdef _WIN32
    FILE* pipe = _popen(full_cmd.c_str(), "r");
#else
    FILE* pipe = popen(full_cmd.c_str(), "r");
#endif

    if (!pipe) {
        exit_code = -1;
        return "Failed to run command";
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }

#ifdef _WIN32
    exit_code = _pclose(pipe);
#else
    exit_code = pclose(pipe);
#endif

    return result;
}

int main(int argc, char* argv[]) {
    std::string exe_path = "digit_recognizer";
    std::string model_path = "digit_model.bin";
    std::string images_dir = "sample_images";
    bool use_gpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--exe" && i + 1 < argc)        exe_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc)  model_path = argv[++i];
        else if (arg == "--images" && i + 1 < argc)  images_dir = argv[++i];
        else if (arg == "--gpu")                     use_gpu = true;
        else if (arg == "--help") {
            std::cout << "Usage: stress_test [options]\n"
                      << "  --exe <path>      Path to digit_recognizer executable\n"
                      << "  --model <path>    Path to trained model file\n"
                      << "  --images <dir>    Path to sample_images directory\n"
                      << "  --gpu             Pass --gpu flag to digit_recognizer\n"
                      << "\nRuns predict in an infinite loop on random images.\n"
                      << "Press Ctrl+C to stop and see summary.\n";
            return 0;
        }
    }

    signal(SIGINT, signal_handler);
#ifdef SIGTERM
    signal(SIGTERM, signal_handler);
#endif

    // Collect image files
    std::vector<std::string> image_files;
    for (auto& entry : fs::directory_iterator(images_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
            image_files.push_back(entry.path().string());
        }
    }

    if (image_files.empty()) {
        std::cerr << "No image files found in " << images_dir << std::endl;
        return 1;
    }

    std::cout << "=== Digit Recognizer Stress Test ===" << std::endl;
    std::cout << "Executable:   " << exe_path << std::endl;
    std::cout << "Model:        " << model_path << std::endl;
    std::cout << "Images dir:   " << images_dir << std::endl;
    std::cout << "Image count:  " << image_files.size() << std::endl;
    std::cout << "GPU:          " << (use_gpu ? "yes" : "no") << std::endl;
    std::cout << "Press Ctrl+C to stop.\n" << std::endl;

    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<size_t> dist(0, image_files.size() - 1);

    Stats stats;
    auto start_time = std::chrono::steady_clock::now();

    while (running) {
        size_t idx = dist(rng);
        const auto& img_path = image_files[idx];
        int expected = extract_expected_digit(img_path);

        std::string cmd = exe_path + " predict " + img_path +
                          " --model " + model_path;
        if (use_gpu) cmd += " --gpu";

        int exit_code = 0;
        std::string output = run_command(cmd, exit_code);

        stats.total++;

        if (exit_code != 0) {
            stats.errors++;
            std::cout << "[" << stats.total << "] ERROR  " << img_path
                      << "  exit_code=" << exit_code << std::endl;
            continue;
        }

        int predicted = parse_predicted_digit(output);

        bool match = (expected >= 0 && predicted == expected);
        if (expected >= 0) {
            if (match) stats.correct++;
            else stats.wrong++;
        }

        const char* status = (expected < 0) ? "????" :
                             match           ? " OK " : "FAIL";

        std::cout << "[" << std::setw(5) << stats.total << "] "
                  << status
                  << "  predicted=" << predicted;
        if (expected >= 0) std::cout << "  expected=" << expected;
        std::cout << "  file=" << fs::path(img_path).filename().string()
                  << std::endl;
    }

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    double secs = std::chrono::duration<double>(elapsed).count();

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Total runs:   " << stats.total << std::endl;
    std::cout << "Correct:      " << stats.correct << std::endl;
    std::cout << "Wrong:        " << stats.wrong << std::endl;
    std::cout << "Errors:       " << stats.errors << std::endl;
    if (stats.correct + stats.wrong > 0) {
        double acc = 100.0 * stats.correct / (stats.correct + stats.wrong);
        std::cout << "Accuracy:     " << std::fixed << std::setprecision(1)
                  << acc << "%" << std::endl;
    }
    std::cout << "Elapsed:      " << std::fixed << std::setprecision(1)
              << secs << "s" << std::endl;
    if (stats.total > 0) {
        std::cout << "Avg per run:  " << std::setprecision(0)
                  << (secs / stats.total * 1000.0) << "ms" << std::endl;
    }

    return 0;
}
