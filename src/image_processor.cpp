#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image_processor.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace digitrec {

ImageProcessor::Image ImageProcessor::load_image(const std::string& filepath) {
    Image img;
    int w, h, c;
    unsigned char* data = stbi_load(filepath.c_str(), &w, &h, &c, 0);
    if (!data) {
        throw std::runtime_error("Failed to load image: " + filepath);
    }

    img.width = w;
    img.height = h;
    img.channels = c;
    img.pixels.assign(data, data + (w * h * c));
    stbi_image_free(data);
    return img;
}

ImageProcessor::Image ImageProcessor::to_grayscale(const Image& img) {
    if (img.channels == 1) return img;

    Image gray;
    gray.width = img.width;
    gray.height = img.height;
    gray.channels = 1;
    gray.pixels.resize(img.width * img.height);

    for (int i = 0; i < img.width * img.height; ++i) {
        int offset = i * img.channels;
        double r = img.pixels[offset];
        double g = img.pixels[offset + 1];
        double b = img.pixels[offset + 2];
        gray.pixels[i] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return gray;
}

ImageProcessor::Image ImageProcessor::resize(const Image& img, int new_width, int new_height) {
    if (img.channels != 1) {
        throw std::runtime_error("resize expects a single-channel (grayscale) image");
    }

    Image resized;
    resized.width = new_width;
    resized.height = new_height;
    resized.channels = 1;
    resized.pixels.resize(new_width * new_height);

    double x_ratio = static_cast<double>(img.width) / new_width;
    double y_ratio = static_cast<double>(img.height) / new_height;

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            double src_x = x * x_ratio;
            double src_y = y * y_ratio;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, img.width - 1);
            int y1 = std::min(y0 + 1, img.height - 1);

            double x_frac = src_x - x0;
            double y_frac = src_y - y0;

            // Bilinear interpolation
            double top = img.pixels[y0 * img.width + x0] * (1.0 - x_frac) +
                         img.pixels[y0 * img.width + x1] * x_frac;
            double bot = img.pixels[y1 * img.width + x0] * (1.0 - x_frac) +
                         img.pixels[y1 * img.width + x1] * x_frac;
            double val = top * (1.0 - y_frac) + bot * y_frac;

            resized.pixels[y * new_width + x] = static_cast<uint8_t>(std::round(val));
        }
    }
    return resized;
}

ImageProcessor::BoundingBox ImageProcessor::find_bounding_box(const Image& img, uint8_t threshold) {
    BoundingBox bb;
    bb.x_min = img.width;
    bb.y_min = img.height;
    bb.x_max = 0;
    bb.y_max = 0;

    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            if (img.pixels[y * img.width + x] > threshold) {
                bb.x_min = std::min(bb.x_min, x);
                bb.y_min = std::min(bb.y_min, y);
                bb.x_max = std::max(bb.x_max, x);
                bb.y_max = std::max(bb.y_max, y);
            }
        }
    }

    if (bb.x_min > bb.x_max) {
        bb = {0, 0, img.width - 1, img.height - 1};
    }
    return bb;
}

ImageProcessor::Image ImageProcessor::center_digit(const Image& img) {
    auto bb = find_bounding_box(img);

    int digit_w = bb.x_max - bb.x_min + 1;
    int digit_h = bb.y_max - bb.y_min + 1;

    // Crop the digit region
    Image cropped;
    cropped.width = digit_w;
    cropped.height = digit_h;
    cropped.channels = 1;
    cropped.pixels.resize(digit_w * digit_h);

    for (int y = 0; y < digit_h; ++y) {
        for (int x = 0; x < digit_w; ++x) {
            cropped.pixels[y * digit_w + x] =
                img.pixels[(bb.y_min + y) * img.width + (bb.x_min + x)];
        }
    }

    // Fit into a 20x20 box preserving aspect ratio, then center in 28x28
    int fit_size = 20;
    double scale = static_cast<double>(fit_size) / std::max(digit_w, digit_h);
    int scaled_w = std::max(1, static_cast<int>(digit_w * scale));
    int scaled_h = std::max(1, static_cast<int>(digit_h * scale));

    auto scaled = resize(cropped, scaled_w, scaled_h);

    Image centered;
    centered.width = TARGET_WIDTH;
    centered.height = TARGET_HEIGHT;
    centered.channels = 1;
    centered.pixels.assign(TARGET_WIDTH * TARGET_HEIGHT, 0);

    int offset_x = (TARGET_WIDTH - scaled_w) / 2;
    int offset_y = (TARGET_HEIGHT - scaled_h) / 2;

    for (int y = 0; y < scaled_h; ++y) {
        for (int x = 0; x < scaled_w; ++x) {
            centered.pixels[(offset_y + y) * TARGET_WIDTH + (offset_x + x)] =
                scaled.pixels[y * scaled_w + x];
        }
    }
    return centered;
}

std::vector<double> ImageProcessor::normalize(const std::vector<uint8_t>& pixels) {
    std::vector<double> normalized(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        normalized[i] = static_cast<double>(pixels[i]) / 255.0;
    }
    return normalized;
}

std::vector<double> ImageProcessor::preprocess(const Image& img) {
    auto gray = to_grayscale(img);

    // MNIST uses white-on-black (bright digit, dark background).
    // Auto-detect: if the border pixels are mostly bright, the image is
    // black-on-white and needs inverting.
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
        Image inverted = gray;
        for (auto& p : inverted.pixels) {
            p = static_cast<uint8_t>(255 - p);
        }
        gray = inverted;
    }

    auto centered = center_digit(gray);
    return normalize(centered.pixels);
}

std::vector<double> ImageProcessor::preprocess_file(const std::string& filepath) {
    auto img = load_image(filepath);
    return preprocess(img);
}

} // namespace digitrec
