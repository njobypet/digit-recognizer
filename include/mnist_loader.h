#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <utility>

namespace digitrec {

class MnistLoader {
public:
    struct Dataset {
        std::vector<std::vector<double>> images;
        std::vector<int> labels;
        std::vector<std::vector<double>> one_hot_labels;
    };

    static Dataset load_training_data(const std::string& directory);
    static Dataset load_test_data(const std::string& directory);

private:
    static std::vector<std::vector<double>> load_images(const std::string& filepath);
    static std::vector<int> load_labels(const std::string& filepath);
    static std::vector<std::vector<double>> to_one_hot(const std::vector<int>& labels,
                                                        int num_classes = 10);
    static uint32_t read_uint32_be(const uint8_t* data);
};

} // namespace digitrec
