#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <functional>

namespace digitrec {

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes, double learning_rate = 0.01);
    ~NeuralNetwork();

    std::vector<double> predict(const std::vector<double>& input) const;
    int predict_digit(const std::vector<double>& input) const;

    void train(const std::vector<double>& input,
               const std::vector<double>& target);

    void train_batch(const std::vector<std::vector<double>>& inputs,
                     const std::vector<std::vector<double>>& targets,
                     int epochs,
                     int batch_size = 32,
                     std::function<void(int, double)> on_epoch = nullptr);

    double evaluate(const std::vector<std::vector<double>>& inputs,
                    const std::vector<int>& labels) const;

    bool save(const std::string& filepath) const;
    bool load(const std::string& filepath);

    void set_learning_rate(double lr) { learning_rate_ = lr; }
    double get_learning_rate() const { return learning_rate_; }

    bool using_gpu() const { return use_gpu_; }
    void enable_gpu(bool enable);

private:
    struct Layer {
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
    };

    std::vector<int> layer_sizes_;
    std::vector<Layer> layers_;
    double learning_rate_;
    mutable std::mt19937 rng_;
    bool use_gpu_ = false;

    void initialize_weights();

    // CPU path
    std::vector<double> forward_layer_cpu(const std::vector<double>& input,
                                          const Layer& layer,
                                          bool apply_relu) const;
    std::vector<double> predict_cpu(const std::vector<double>& input) const;
    void train_cpu(const std::vector<double>& input, const std::vector<double>& target);

#ifdef USE_HIP
    struct GpuLayer {
        double* d_weights = nullptr;
        double* d_biases = nullptr;
        int rows = 0;
        int cols = 0;
    };
    std::vector<GpuLayer> gpu_layers_;

    // Pre-allocated scratch buffers for train_gpu / predict_gpu to avoid
    // repeated hipMalloc/hipFree per sample (which corrupts the heap).
    struct GpuScratch {
        std::vector<double*> activations;    // num_layers + 1
        std::vector<double*> pre_activations; // num_layers
        std::vector<double*> deltas;         // num_layers
        double* target = nullptr;
        bool allocated = false;
    };
    mutable GpuScratch scratch_;

    void upload_weights_to_gpu();
    void download_weights_from_gpu();
    void free_gpu_layers();
    void alloc_scratch();
    void free_scratch();

    std::vector<double> predict_gpu(const std::vector<double>& input) const;
    void train_gpu(const std::vector<double>& input, const std::vector<double>& target);
#endif

    static std::vector<double> relu(const std::vector<double>& x);
    static std::vector<double> relu_derivative(const std::vector<double>& x);
    static std::vector<double> softmax(const std::vector<double>& x);
};

} // namespace digitrec
