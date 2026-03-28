#include "neural_network.h"
#include "gpu_backend.h"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <cassert>
#include <chrono>
#include <limits>

namespace digitrec {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes, double learning_rate)
    : layer_sizes_(layer_sizes)
    , learning_rate_(learning_rate)
    , rng_(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()))
{
    initialize_weights();
}

NeuralNetwork::~NeuralNetwork() {
#ifdef USE_HIP
    free_gpu_layers();
#endif
}

void NeuralNetwork::enable_gpu(bool enable) {
#ifdef USE_HIP
    auto& gpu = GpuBackend::instance();
    if (enable && gpu.is_available()) {
        use_gpu_ = true;
        upload_weights_to_gpu();
        std::cout << "[GPU] Neural network GPU acceleration ENABLED" << std::endl;
    } else {
        if (use_gpu_) {
            download_weights_from_gpu();
            free_gpu_layers();
        }
        use_gpu_ = false;
        if (enable) {
            std::cout << "[GPU] GPU requested but not available. Using CPU." << std::endl;
        }
    }
#else
    (void)enable;
    use_gpu_ = false;
    if (enable) {
        std::cout << "[GPU] Built without HIP support. Using CPU." << std::endl;
    }
#endif
}

void NeuralNetwork::initialize_weights() {
    layers_.clear();
    for (size_t i = 0; i + 1 < layer_sizes_.size(); ++i) {
        Layer layer;
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];

        double stddev = std::sqrt(2.0 / fan_in);
        std::normal_distribution<double> dist(0.0, stddev);

        layer.weights.resize(fan_out, std::vector<double>(fan_in));
        layer.biases.resize(fan_out, 0.0);

        for (int r = 0; r < fan_out; ++r) {
            for (int c = 0; c < fan_in; ++c) {
                layer.weights[r][c] = dist(rng_);
            }
        }
        layers_.push_back(std::move(layer));
    }
}

// ============================================================================
// Static helpers
// ============================================================================

std::vector<double> NeuralNetwork::relu(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::max(0.0, x[i]);
    }
    return result;
}

std::vector<double> NeuralNetwork::relu_derivative(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0.0 ? 1.0 : 0.0;
    }
    return result;
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    double max_val = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    for (auto& v : result) {
        v /= sum;
    }
    return result;
}

// ============================================================================
// CPU path
// ============================================================================

std::vector<double> NeuralNetwork::forward_layer_cpu(const std::vector<double>& input,
                                                      const Layer& layer,
                                                      bool apply_relu) const {
    int out_size = static_cast<int>(layer.biases.size());
    std::vector<double> output(out_size);

    for (int i = 0; i < out_size; ++i) {
        double sum = layer.biases[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += layer.weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    if (apply_relu) {
        output = relu(output);
    }
    return output;
}

std::vector<double> NeuralNetwork::predict_cpu(const std::vector<double>& input) const {
    OpLog::cpu_phase("--- CPU predict: begin forward pass ---");
    OpLog::cpu("Input vector allocated on CPU", input.size() * sizeof(double));

    std::vector<double> current = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        bool is_last = (i == layers_.size() - 1);
        int out_size = static_cast<int>(layers_[i].biases.size());
        int in_size = static_cast<int>(current.size());

        std::string layer_label = "Layer " + std::to_string(i) + " (" +
            std::to_string(in_size) + " -> " + std::to_string(out_size) + ")";
        OpLog::cpu_phase(("  " + layer_label + ": computing matvec on CPU...").c_str());
        OpLog::cpu(("  " + layer_label + ": output buffer allocated on CPU").c_str(),
                   out_size * sizeof(double));

        current = forward_layer_cpu(current, layers_[i], !is_last);

        if (!is_last) {
            OpLog::cpu_phase(("  " + layer_label + ": applied ReLU activation").c_str());
        }
    }

    OpLog::cpu_phase("  Output layer: computing softmax on CPU...");
    auto result = softmax(current);
    OpLog::cpu_phase("--- CPU predict: forward pass complete ---");
    return result;
}

void NeuralNetwork::train_cpu(const std::vector<double>& input,
                               const std::vector<double>& target) {
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> pre_activations;
    activations.push_back(input);

    std::vector<double> current = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        bool is_last = (i == layers_.size() - 1);
        int out_size = static_cast<int>(layers_[i].biases.size());
        std::vector<double> z(out_size);

        for (int r = 0; r < out_size; ++r) {
            double sum = layers_[i].biases[r];
            for (size_t c = 0; c < current.size(); ++c) {
                sum += layers_[i].weights[r][c] * current[c];
            }
            z[r] = sum;
        }
        pre_activations.push_back(z);

        if (is_last) {
            current = softmax(z);
        } else {
            current = relu(z);
        }
        activations.push_back(current);
    }

    size_t num_layers = layers_.size();
    std::vector<std::vector<double>> deltas(num_layers);

    {
        auto& output = activations.back();
        deltas[num_layers - 1].resize(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            deltas[num_layers - 1][i] = output[i] - target[i];
        }
    }

    for (int l = static_cast<int>(num_layers) - 2; l >= 0; --l) {
        auto& z = pre_activations[l];
        auto rd = relu_derivative(z);
        int current_size = static_cast<int>(z.size());
        int next_size = static_cast<int>(deltas[l + 1].size());

        deltas[l].resize(current_size, 0.0);
        for (int i = 0; i < current_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < next_size; ++j) {
                sum += layers_[l + 1].weights[j][i] * deltas[l + 1][j];
            }
            deltas[l][i] = sum * rd[i];
        }
    }

    for (size_t l = 0; l < num_layers; ++l) {
        auto& layer = layers_[l];
        auto& delta = deltas[l];
        auto& prev_act = activations[l];

        for (size_t i = 0; i < delta.size(); ++i) {
            for (size_t j = 0; j < prev_act.size(); ++j) {
                layer.weights[i][j] -= learning_rate_ * delta[i] * prev_act[j];
            }
            layer.biases[i] -= learning_rate_ * delta[i];
        }
    }
}

// ============================================================================
// GPU path
// ============================================================================

#ifdef USE_HIP

void NeuralNetwork::upload_weights_to_gpu() {
    free_gpu_layers();
    auto& gpu = GpuBackend::instance();

    gpu_layers_.resize(layers_.size());
    for (size_t l = 0; l < layers_.size(); ++l) {
        int rows = static_cast<int>(layers_[l].biases.size());
        int cols = static_cast<int>(layers_[l].weights[0].size());

        gpu_layers_[l].rows = rows;
        gpu_layers_[l].cols = cols;

        // Flatten row-major weight matrix
        std::vector<double> flat(rows * cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                flat[r * cols + c] = layers_[l].weights[r][c];
            }
        }

        gpu_layers_[l].d_weights = gpu.alloc(rows * cols);
        gpu_layers_[l].d_biases = gpu.alloc(rows);
        gpu.copy_to_device(gpu_layers_[l].d_weights, flat.data(), rows * cols);
        gpu.copy_to_device(gpu_layers_[l].d_biases, layers_[l].biases.data(), rows);
    }
    std::cout << "[GPU] Uploaded " << layers_.size() << " layers to device memory" << std::endl;
}

void NeuralNetwork::download_weights_from_gpu() {
    auto& gpu = GpuBackend::instance();

    for (size_t l = 0; l < gpu_layers_.size(); ++l) {
        int rows = gpu_layers_[l].rows;
        int cols = gpu_layers_[l].cols;

        std::vector<double> flat(rows * cols);
        gpu.copy_to_host(flat.data(), gpu_layers_[l].d_weights, rows * cols);
        gpu.copy_to_host(layers_[l].biases.data(), gpu_layers_[l].d_biases, rows);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                layers_[l].weights[r][c] = flat[r * cols + c];
            }
        }
    }
}

void NeuralNetwork::free_gpu_layers() {
    auto& gpu = GpuBackend::instance();
    for (auto& gl : gpu_layers_) {
        gpu.free(gl.d_weights);
        gpu.free(gl.d_biases);
        gl.d_weights = nullptr;
        gl.d_biases = nullptr;
    }
    gpu_layers_.clear();
}

std::vector<double> NeuralNetwork::predict_gpu(const std::vector<double>& input) const {
    auto& gpu = GpuBackend::instance();
    size_t num_layers = gpu_layers_.size();

    OpLog::gpu_phase("--- GPU predict: begin forward pass ---");

    OpLog::gpu_phase("Allocating GPU memory for input vector...");
    double* d_input = gpu.alloc(input.size());
    OpLog::gpu_phase("Copying input data from CPU to GPU...");
    gpu.copy_to_device(d_input, input.data(), input.size());

    double* d_current = d_input;
    std::vector<double*> temp_buffers;

    for (size_t l = 0; l < num_layers; ++l) {
        bool is_last = (l == num_layers - 1);
        int rows = gpu_layers_[l].rows;
        int cols = gpu_layers_[l].cols;

        std::string layer_label = "Layer " + std::to_string(l) + " (" +
            std::to_string(cols) + " -> " + std::to_string(rows) + ")";

        OpLog::gpu_phase(("  " + layer_label + ": allocating output buffer on GPU...").c_str());
        double* d_output = gpu.alloc(rows);
        temp_buffers.push_back(d_output);

        OpLog::gpu_phase(("  " + layer_label + ": launching matvec kernel...").c_str());
        gpu.matvec_multiply(gpu_layers_[l].d_weights, d_current, d_output,
                            gpu_layers_[l].d_biases, rows, cols);

        if (!is_last) {
            OpLog::gpu_phase(("  " + layer_label + ": launching ReLU kernel...").c_str());
            gpu.relu_forward(d_output, rows);
        } else {
            OpLog::gpu_phase(("  " + layer_label + ": launching softmax kernel...").c_str());
            gpu.softmax_forward(d_output, rows);
        }

        d_current = d_output;
    }

    OpLog::gpu_phase("Synchronizing GPU (waiting for all kernels to complete)...");
    gpu.synchronize();

    int out_size = gpu_layers_.back().rows;
    OpLog::cpu("Allocating result buffer on CPU", out_size * sizeof(double));
    std::vector<double> result(out_size);
    OpLog::gpu_phase("Copying result data from GPU to CPU...");
    gpu.copy_to_host(result.data(), d_current, out_size);

    OpLog::gpu_phase("Freeing temporary GPU memory...");
    gpu.free(d_input);
    for (auto* p : temp_buffers) gpu.free(p);

    OpLog::gpu_phase("--- GPU predict: forward pass complete ---");
    return result;
}

void NeuralNetwork::train_gpu(const std::vector<double>& input,
                               const std::vector<double>& target) {
    auto& gpu = GpuBackend::instance();
    size_t num_layers = gpu_layers_.size();

    // ---- Forward pass: keep all activations and pre-activations on device ----
    std::vector<double*> d_activations(num_layers + 1);
    std::vector<double*> d_pre_activations(num_layers);

    d_activations[0] = gpu.alloc(input.size());
    gpu.copy_to_device(d_activations[0], input.data(), input.size());

    for (size_t l = 0; l < num_layers; ++l) {
        bool is_last = (l == num_layers - 1);
        int rows = gpu_layers_[l].rows;
        int cols = gpu_layers_[l].cols;

        d_pre_activations[l] = gpu.alloc(rows);
        d_activations[l + 1] = gpu.alloc(rows);

        // z = W*a + b
        gpu.matvec_multiply(gpu_layers_[l].d_weights, d_activations[l],
                            d_pre_activations[l], gpu_layers_[l].d_biases, rows, cols);

        // Copy pre-activation to activation buffer, then apply activation in-place
        gpu.copy_on_device(d_activations[l + 1], d_pre_activations[l], rows);

        if (is_last) {
            gpu.softmax_forward(d_activations[l + 1], rows);
        } else {
            gpu.relu_forward(d_activations[l + 1], rows);
        }
    }

    // ---- Backward pass ----
    std::vector<double*> d_deltas(num_layers);

    // Output delta
    int out_size = gpu_layers_[num_layers - 1].rows;
    d_deltas[num_layers - 1] = gpu.alloc(out_size);

    double* d_target = gpu.alloc(target.size());
    gpu.copy_to_device(d_target, target.data(), target.size());

    gpu.compute_output_delta(d_activations[num_layers], d_target,
                             d_deltas[num_layers - 1], out_size);
    gpu.free(d_target);

    // Hidden layer deltas
    for (int l = static_cast<int>(num_layers) - 2; l >= 0; --l) {
        int current_size = gpu_layers_[l].rows;
        int next_size = gpu_layers_[l + 1].rows;

        d_deltas[l] = gpu.alloc(current_size);
        gpu.backprop_delta(gpu_layers_[l + 1].d_weights, d_deltas[l + 1],
                           d_pre_activations[l], d_deltas[l],
                           current_size, next_size);
    }

    // ---- Weight updates ----
    for (size_t l = 0; l < num_layers; ++l) {
        int rows = gpu_layers_[l].rows;
        int cols = gpu_layers_[l].cols;
        gpu.update_weights(gpu_layers_[l].d_weights, gpu_layers_[l].d_biases,
                           d_deltas[l], d_activations[l],
                           rows, cols, learning_rate_);
    }

    gpu.synchronize();

    // ---- Free temporary device memory ----
    for (size_t l = 0; l <= num_layers; ++l) gpu.free(d_activations[l]);
    for (size_t l = 0; l < num_layers; ++l) gpu.free(d_pre_activations[l]);
    for (size_t l = 0; l < num_layers; ++l) gpu.free(d_deltas[l]);
}

#endif // USE_HIP

// ============================================================================
// Public dispatch (GPU or CPU)
// ============================================================================

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) const {
#ifdef USE_HIP
    if (use_gpu_) return predict_gpu(input);
#endif
    return predict_cpu(input);
}

int NeuralNetwork::predict_digit(const std::vector<double>& input) const {
    auto probs = predict(input);
    return static_cast<int>(
        std::max_element(probs.begin(), probs.end()) - probs.begin());
}

void NeuralNetwork::train(const std::vector<double>& input,
                           const std::vector<double>& target) {
#ifdef USE_HIP
    if (use_gpu_) { train_gpu(input, target); return; }
#endif
    train_cpu(input, target);
}

void NeuralNetwork::train_batch(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& targets,
                                 int epochs,
                                 int batch_size,
                                 std::function<void(int, double)> on_epoch) {
    assert(inputs.size() == targets.size());
    size_t n = inputs.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng_);
        double total_loss = 0.0;
#ifdef USE_HIP
        bool first_batch = true;
#endif

        for (size_t start = 0; start < n; start += batch_size) {
            size_t end = std::min(start + static_cast<size_t>(batch_size), n);

#ifdef USE_HIP
            if (use_gpu_) {
                GpuBackend::instance().set_kernel_logging(first_batch);
                first_batch = false;
            }
#endif

            for (size_t i = start; i < end; ++i) {
                size_t idx = indices[i];
                train(inputs[idx], targets[idx]);

                auto pred = predict(inputs[idx]);
                for (size_t k = 0; k < targets[idx].size(); ++k) {
                    if (targets[idx][k] > 0.5) {
                        total_loss -= std::log(std::max(pred[k], 1e-10));
                    }
                }
            }
        }

#ifdef USE_HIP
        if (use_gpu_) {
            GpuBackend::instance().set_kernel_logging(true);
            download_weights_from_gpu();
        }
#endif

        if (on_epoch) {
            on_epoch(epoch, total_loss / static_cast<double>(n));
        }

#ifdef USE_HIP
        if (use_gpu_) {
            upload_weights_to_gpu();
        }
#endif
    }

#ifdef USE_HIP
    if (use_gpu_) {
        download_weights_from_gpu();
    }
#endif
}

double NeuralNetwork::evaluate(const std::vector<std::vector<double>>& inputs,
                                const std::vector<int>& labels) const {
    assert(inputs.size() == labels.size());
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (predict_digit(inputs[i]) == labels[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(inputs.size());
}

bool NeuralNetwork::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic = 0x44524E4E;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    uint32_t num_sizes = static_cast<uint32_t>(layer_sizes_.size());
    file.write(reinterpret_cast<const char*>(&num_sizes), sizeof(num_sizes));
    for (auto s : layer_sizes_) {
        int32_t val = static_cast<int32_t>(s);
        file.write(reinterpret_cast<const char*>(&val), sizeof(val));
    }

    file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));

    for (auto& layer : layers_) {
        for (auto& row : layer.weights) {
            file.write(reinterpret_cast<const char*>(row.data()),
                       static_cast<std::streamsize>(row.size() * sizeof(double)));
        }
        file.write(reinterpret_cast<const char*>(layer.biases.data()),
                   static_cast<std::streamsize>(layer.biases.size() * sizeof(double)));
    }

    return file.good();
}

bool NeuralNetwork::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x44524E4E) return false;

    uint32_t num_sizes = 0;
    file.read(reinterpret_cast<char*>(&num_sizes), sizeof(num_sizes));

    layer_sizes_.resize(num_sizes);
    for (uint32_t i = 0; i < num_sizes; ++i) {
        int32_t val = 0;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
        layer_sizes_[i] = val;
    }

    file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));

    layers_.clear();
    for (size_t i = 0; i + 1 < layer_sizes_.size(); ++i) {
        Layer layer;
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];

        layer.weights.resize(fan_out, std::vector<double>(fan_in));
        layer.biases.resize(fan_out);

        for (int r = 0; r < fan_out; ++r) {
            file.read(reinterpret_cast<char*>(layer.weights[r].data()),
                      static_cast<std::streamsize>(fan_in * sizeof(double)));
        }
        file.read(reinterpret_cast<char*>(layer.biases.data()),
                  static_cast<std::streamsize>(fan_out * sizeof(double)));

        layers_.push_back(std::move(layer));
    }

    return file.good();
}

} // namespace digitrec
