#pragma once

#include "neural_network.h"
#include "image_processor.h"
#include <string>
#include <vector>
#include <utility>

namespace digitrec {

struct PredictionResult {
    int digit;
    double confidence;
    std::vector<double> probabilities;
};

class DigitRecognizer {
public:
    DigitRecognizer();
    ~DigitRecognizer();

    bool init_gpu();

    PredictionResult recognize(const std::string& image_path) const;
    PredictionResult recognize(const std::vector<double>& preprocessed_input) const;

    std::string recognize_multi_digit(const std::string& image_path) const;

    void train_on_mnist(const std::string& mnist_dir,
                        int epochs = 10,
                        int batch_size = 32);

    double test_on_mnist(const std::string& mnist_dir) const;

    bool save_model(const std::string& filepath) const;
    bool load_model(const std::string& filepath);

    bool is_model_loaded() const { return model_loaded_; }
    bool using_gpu() const { return network_.using_gpu(); }

private:
    NeuralNetwork network_;
    bool model_loaded_;

    static std::vector<int> default_architecture();
};

} // namespace digitrec
