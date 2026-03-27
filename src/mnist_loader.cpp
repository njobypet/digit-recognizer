#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace digitrec {

uint32_t MnistLoader::read_uint32_be(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8)  |
           (static_cast<uint32_t>(data[3]));
}

std::vector<std::vector<double>> MnistLoader::load_images(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST image file: " + filepath);
    }

    uint8_t header[16];
    file.read(reinterpret_cast<char*>(header), 16);

    uint32_t magic = read_uint32_be(header);
    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }

    uint32_t num_images = read_uint32_be(header + 4);
    uint32_t rows = read_uint32_be(header + 8);
    uint32_t cols = read_uint32_be(header + 12);
    uint32_t image_size = rows * cols;

    std::vector<std::vector<double>> images(num_images);
    std::vector<uint8_t> buffer(image_size);

    for (uint32_t i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        images[i].resize(image_size);
        for (uint32_t j = 0; j < image_size; ++j) {
            images[i][j] = static_cast<double>(buffer[j]) / 255.0;
        }
    }

    std::cout << "Loaded " << num_images << " images (" << rows << "x" << cols << ")" << std::endl;
    return images;
}

std::vector<int> MnistLoader::load_labels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open MNIST label file: " + filepath);
    }

    uint8_t header[8];
    file.read(reinterpret_cast<char*>(header), 8);

    uint32_t magic = read_uint32_be(header);
    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }

    uint32_t num_labels = read_uint32_be(header + 4);
    std::vector<uint8_t> raw(num_labels);
    file.read(reinterpret_cast<char*>(raw.data()), num_labels);

    std::vector<int> labels(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        labels[i] = static_cast<int>(raw[i]);
    }

    std::cout << "Loaded " << num_labels << " labels" << std::endl;
    return labels;
}

std::vector<std::vector<double>> MnistLoader::to_one_hot(const std::vector<int>& labels,
                                                          int num_classes) {
    std::vector<std::vector<double>> one_hot(labels.size(),
                                              std::vector<double>(num_classes, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        one_hot[i][labels[i]] = 1.0;
    }
    return one_hot;
}

MnistLoader::Dataset MnistLoader::load_training_data(const std::string& directory) {
    std::string sep = "/";
#ifdef _WIN32
    if (directory.find('\\') != std::string::npos) {
        sep = "\\";
    }
#endif
    std::string base = directory;
    if (!base.empty() && base.back() != '/' && base.back() != '\\') {
        base += sep;
    }

    Dataset ds;
    ds.images = load_images(base + "train-images-idx3-ubyte");
    ds.labels = load_labels(base + "train-labels-idx1-ubyte");
    ds.one_hot_labels = to_one_hot(ds.labels);
    return ds;
}

MnistLoader::Dataset MnistLoader::load_test_data(const std::string& directory) {
    std::string sep = "/";
#ifdef _WIN32
    if (directory.find('\\') != std::string::npos) {
        sep = "\\";
    }
#endif
    std::string base = directory;
    if (!base.empty() && base.back() != '/' && base.back() != '\\') {
        base += sep;
    }

    Dataset ds;
    ds.images = load_images(base + "t10k-images-idx3-ubyte");
    ds.labels = load_labels(base + "t10k-labels-idx1-ubyte");
    ds.one_hot_labels = to_one_hot(ds.labels);
    return ds;
}

} // namespace digitrec
