#include "digit_recognizer.h"
#include "gpu_backend.h"
#include "mnist_loader.h"
#include <iostream>
#include <algorithm>
#include <iomanip>

namespace digitrec {

std::vector<int> DigitRecognizer::default_architecture() {
    return {784, 256, 128, 10};
}

DigitRecognizer::DigitRecognizer()
    : network_(default_architecture(), 0.005)
    , model_loaded_(false)
{
}

DigitRecognizer::~DigitRecognizer() = default;

bool DigitRecognizer::init_gpu() {
    auto& gpu = GpuBackend::instance();
    if (!gpu.is_available()) {
        if (!gpu.initialize()) {
            return false;
        }
    }
    network_.enable_gpu(true);
    return network_.using_gpu();
}

PredictionResult DigitRecognizer::recognize(const std::string& image_path) const {
    OpLog::cpu_phase("=== Loading and preprocessing image ===");
    OpLog::cpu_phase(("Image path: " + image_path).c_str());

    OpLog::cpu_phase("Loading image from disk...");
    auto img = ImageProcessor::load_image(image_path);
    OpLog::cpu_phase(("Loaded: " + std::to_string(img.width) + "x" + std::to_string(img.height) +
                      " pixels, " + std::to_string(img.channels) + " channels").c_str());
    OpLog::cpu("Raw image pixel buffer allocated on CPU",
               img.width * img.height * img.channels);

    OpLog::cpu_phase("Preprocessing: grayscale, auto-invert, center, normalize...");
    auto input = ImageProcessor::preprocess(img);
    OpLog::cpu("Preprocessed input vector (28x28 = 784 doubles) on CPU",
               input.size() * sizeof(double));
    OpLog::cpu_phase("=== Preprocessing complete ===");

    return recognize(input);
}

PredictionResult DigitRecognizer::recognize(const std::vector<double>& preprocessed_input) const {
    OpLog::phase("=== Running neural network inference ===");
    OpLog::phase(using_gpu() ? "Compute path: GPU (ROCm/HIP)" : "Compute path: CPU");

    PredictionResult result;
    result.probabilities = network_.predict(preprocessed_input);
    auto it = std::max_element(result.probabilities.begin(), result.probabilities.end());
    result.digit = static_cast<int>(it - result.probabilities.begin());
    result.confidence = *it;

    OpLog::phase(("=== Inference complete: digit=" + std::to_string(result.digit) +
                  " confidence=" + std::to_string(result.confidence * 100.0).substr(0, 5) +
                  "% ===").c_str());
    return result;
}

std::string DigitRecognizer::recognize_multi_digit(const std::string& image_path) const {
    auto img = ImageProcessor::load_image(image_path);
    auto gray = ImageProcessor::to_grayscale(img);

    // Auto-invert if the background is bright (dark-on-light input)
    long border_sum = 0;
    int border_count = 0;
    for (int x = 0; x < gray.width; ++x) {
        border_sum += gray.pixels[x];
        border_sum += gray.pixels[(gray.height - 1) * gray.width + x];
        border_count += 2;
    }
    for (int y = 1; y < gray.height - 1; ++y) {
        border_sum += gray.pixels[y * gray.width];
        border_sum += gray.pixels[y * gray.width + (gray.width - 1)];
        border_count += 2;
    }
    double avg_border = static_cast<double>(border_sum) / border_count;
    if (avg_border > 128.0) {
        for (auto& p : gray.pixels) {
            p = static_cast<uint8_t>(255 - p);
        }
    }

    // Column projection: count pixels above brightness threshold per column
    uint8_t ink_threshold = 30;
    std::vector<int> col_counts(gray.width, 0);
    for (int y = 0; y < gray.height; ++y) {
        for (int x = 0; x < gray.width; ++x) {
            if (gray.pixels[y * gray.width + x] > ink_threshold) {
                col_counts[x]++;
            }
        }
    }

    // A column is "ink" if it has at least 1 bright pixel
    // Find raw runs of ink columns
    std::vector<std::pair<int, int>> raw_segments;
    int seg_start = -1;
    for (int x = 0; x < gray.width; ++x) {
        if (col_counts[x] > 0) {
            if (seg_start < 0) seg_start = x;
        } else {
            if (seg_start >= 0) {
                raw_segments.emplace_back(seg_start, x);
                seg_start = -1;
            }
        }
    }
    if (seg_start >= 0) {
        raw_segments.emplace_back(seg_start, gray.width);
    }

    if (raw_segments.empty()) {
        raw_segments.emplace_back(0, gray.width);
    }

    // Merge segments separated by very small gaps (thin strokes can create
    // within-digit blank columns from anti-aliasing). Only merge gaps that
    // are tiny compared to the segments they separate.
    // Use a fraction of average segment width to decide.
    int total_ink_width = 0;
    for (auto& s : raw_segments) total_ink_width += (s.second - s.first);
    int avg_seg_width = (raw_segments.size() > 0)
        ? total_ink_width / static_cast<int>(raw_segments.size()) : 1;
    int min_gap = std::max(2, std::min(avg_seg_width / 3, gray.width / 60));
    std::vector<std::pair<int, int>> merged;
    merged.push_back(raw_segments[0]);
    for (size_t i = 1; i < raw_segments.size(); ++i) {
        int gap = raw_segments[i].first - merged.back().second;
        if (gap < min_gap) {
            merged.back().second = raw_segments[i].second;
        } else {
            merged.push_back(raw_segments[i]);
        }
    }

    // Filter out tiny noise fragments (< 2% of image width)
    int min_width = std::max(3, gray.width / 50);
    std::vector<std::pair<int, int>> filtered;
    for (auto& seg : merged) {
        if (seg.second - seg.first >= min_width) {
            filtered.push_back(seg);
        }
    }
    if (filtered.empty()) filtered = merged;

    std::string result;
    for (auto& [x_start, x_end] : filtered) {
        ImageProcessor::Image segment;
        segment.width = x_end - x_start;
        segment.height = gray.height;
        segment.channels = 1;
        segment.pixels.resize(segment.width * segment.height);

        for (int y = 0; y < segment.height; ++y) {
            for (int x = 0; x < segment.width; ++x) {
                segment.pixels[y * segment.width + x] =
                    gray.pixels[y * gray.width + (x_start + x)];
            }
        }

        // center_digit handles cropping and normalizing internally
        auto centered = ImageProcessor::center_digit(segment);
        auto input = ImageProcessor::normalize(centered.pixels);
        auto pred = recognize(input);
        result += std::to_string(pred.digit);
    }

    return result;
}

void DigitRecognizer::train_on_mnist(const std::string& mnist_dir,
                                      int epochs,
                                      int batch_size) {
    std::cout << "Loading MNIST training data from: " << mnist_dir << std::endl;
    auto train_data = MnistLoader::load_training_data(mnist_dir);

    std::cout << "Training network..." << std::endl;
    std::cout << "Architecture: 784 -> 256 -> 128 -> 10" << std::endl;
    std::cout << "Learning rate: " << network_.get_learning_rate() << std::endl;
    std::cout << "Compute device: " << (using_gpu() ? "AMD GPU (ROCm/HIP)" : "CPU") << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    network_.train_batch(
        train_data.images,
        train_data.one_hot_labels,
        epochs,
        batch_size,
        [&](int epoch, double loss) {
            double acc = network_.evaluate(train_data.images, train_data.labels);
            std::cout << "Epoch " << std::setw(3) << (epoch + 1)
                      << "/" << epochs
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss
                      << " | Accuracy: " << std::setprecision(2) << (acc * 100.0) << "%"
                      << std::endl;
        }
    );

    model_loaded_ = true;
    std::cout << "Training complete." << std::endl;
}

double DigitRecognizer::test_on_mnist(const std::string& mnist_dir) const {
    std::cout << "Loading MNIST test data from: " << mnist_dir << std::endl;
    auto test_data = MnistLoader::load_test_data(mnist_dir);

    double accuracy = network_.evaluate(test_data.images, test_data.labels);
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(2)
              << (accuracy * 100.0) << "%" << std::endl;
    return accuracy;
}

bool DigitRecognizer::save_model(const std::string& filepath) const {
    if (network_.save(filepath)) {
        std::cout << "Model saved to: " << filepath << std::endl;
        return true;
    }
    std::cerr << "Failed to save model to: " << filepath << std::endl;
    return false;
}

bool DigitRecognizer::load_model(const std::string& filepath) {
    if (network_.load(filepath)) {
        model_loaded_ = true;
        std::cout << "Model loaded from: " << filepath << std::endl;
        return true;
    }
    std::cerr << "Failed to load model from: " << filepath << std::endl;
    return false;
}

} // namespace digitrec
