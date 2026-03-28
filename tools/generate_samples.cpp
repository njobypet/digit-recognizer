#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <map>

uint32_t read_u32_be(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8)  | uint32_t(p[3]);
}

void write_le16(std::ofstream& f, uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); }
void write_le32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); }

bool write_bmp(const std::string& path, const uint8_t* pixels, int w, int h) {
    int row_bytes = (w * 3 + 3) & ~3;
    uint32_t pixel_size = row_bytes * h;
    uint32_t file_size = 54 + pixel_size;

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // BMP header
    f.put('B'); f.put('M');
    write_le32(f, file_size);
    write_le16(f, 0); write_le16(f, 0);
    write_le32(f, 54);

    // DIB header
    write_le32(f, 40);
    write_le32(f, w);
    write_le32(f, h);
    write_le16(f, 1);
    write_le16(f, 24);
    write_le32(f, 0);
    write_le32(f, pixel_size);
    write_le32(f, 2835); write_le32(f, 2835);
    write_le32(f, 0); write_le32(f, 0);

    // BMP stores rows bottom-to-top
    std::vector<uint8_t> row(row_bytes, 0);
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            uint8_t v = pixels[y * w + x];
            row[x * 3 + 0] = v;
            row[x * 3 + 1] = v;
            row[x * 3 + 2] = v;
        }
        f.write(reinterpret_cast<char*>(row.data()), row_bytes);
    }
    return f.good();
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    std::string out_dir = "sample_images";
    int per_digit = 10;

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) out_dir = argv[2];

    std::string img_path = data_dir + "/train-images-idx3-ubyte";
    std::string lbl_path = data_dir + "/train-labels-idx1-ubyte";

    // On Windows, try backslash if forward slash fails
    std::ifstream img_file(img_path, std::ios::binary);
    if (!img_file) {
        img_path = data_dir + "\\train-images-idx3-ubyte";
        lbl_path = data_dir + "\\train-labels-idx1-ubyte";
        img_file.open(img_path, std::ios::binary);
    }
    if (!img_file) {
        std::cerr << "Cannot open " << img_path << std::endl;
        return 1;
    }

    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file) {
        std::cerr << "Cannot open " << lbl_path << std::endl;
        return 1;
    }

    // Read image header
    uint8_t ih[16], lh[8];
    img_file.read(reinterpret_cast<char*>(ih), 16);
    lbl_file.read(reinterpret_cast<char*>(lh), 8);

    uint32_t num = read_u32_be(ih + 4);
    uint32_t rows = read_u32_be(ih + 8);
    uint32_t cols = read_u32_be(ih + 12);
    uint32_t img_size = rows * cols;

    std::cout << "MNIST: " << num << " images, " << rows << "x" << cols << std::endl;

    std::map<int, int> counts;
    std::vector<uint8_t> buf(img_size);
    int total = 0;

    for (uint32_t i = 0; i < num && total < per_digit * 10; ++i) {
        img_file.read(reinterpret_cast<char*>(buf.data()), img_size);
        uint8_t label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        int d = label;

        if (counts[d] >= per_digit) continue;

        std::string sep = "/";
#ifdef _WIN32
        sep = "\\";
#endif
        std::string filename = out_dir + sep + "digit_" +
            std::to_string(d) + "_" + std::to_string(counts[d]) + ".bmp";

        if (write_bmp(filename, buf.data(), cols, rows)) {
            std::cout << "  Wrote " << filename << "  (digit " << d << ")" << std::endl;
            counts[d]++;
            total++;
        } else {
            std::cerr << "  Failed to write " << filename << std::endl;
        }
    }

    std::cout << "\nGenerated " << total << " sample images in " << out_dir << std::endl;
    return 0;
}
