#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace digitrec {

class ImageProcessor {
public:
    static constexpr int TARGET_WIDTH = 28;
    static constexpr int TARGET_HEIGHT = 28;
    static constexpr int INPUT_SIZE = TARGET_WIDTH * TARGET_HEIGHT;

    struct Image {
        std::vector<uint8_t> pixels;
        int width = 0;
        int height = 0;
        int channels = 0;
    };

    static Image load_image(const std::string& filepath);

    static std::vector<double> preprocess(const Image& img);

    static std::vector<double> preprocess_file(const std::string& filepath);

    static Image to_grayscale(const Image& img);

    static Image resize(const Image& img, int new_width, int new_height);

    static std::vector<double> normalize(const std::vector<uint8_t>& pixels);

    static Image center_digit(const Image& img);

private:
    struct BoundingBox {
        int x_min, y_min, x_max, y_max;
    };

    static BoundingBox find_bounding_box(const Image& img, uint8_t threshold = 30);
};

} // namespace digitrec
