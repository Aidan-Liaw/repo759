#include "file_handling.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>

std::vector<unsigned char> read_file(const std::string &image_path) {
    std::ifstream file(image_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open input JPEG: " + image_path);
    }

    const std::streamsize file_size = file.tellg();
    if (file_size <= 0) {
        throw std::runtime_error("Input JPEG is empty: " + image_path);
    }

    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> jpeg_buffer(static_cast<std::size_t>(file_size));
    if (!file.read(reinterpret_cast<char *>(jpeg_buffer.data()), file_size)) {
        throw std::runtime_error("Could not read input JPEG: " + image_path);
    }

    return jpeg_buffer;
}

std::vector<float> make_gaussian_mask(int R, float sigma) {
    const int mask_length = 2 * R + 1;
    std::vector<float> mask(static_cast<std::size_t>(mask_length));

    if (sigma <= 0.0f) {
        sigma = std::max(1.0f, static_cast<float>(R) * 0.5f);
    }

    const float inv_two_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    float sum = 0.0f;

    for (int idx = 0; idx < mask_length; ++idx) {
        const int offset = idx - R;
        const float value = std::exp(-static_cast<float>(offset * offset) * inv_two_sigma_sq);
        mask[static_cast<std::size_t>(idx)] = value;
        sum += value;
    }

    for (int idx = 0; idx < mask_length; ++idx) {
        mask[static_cast<std::size_t>(idx)] /= sum;
    }

    return mask;
}


std::string default_output_image_path(const std::string &image_path, int channels) {
    std::filesystem::path input_path(image_path);
    std::filesystem::path stem = input_path.stem();
    if (stem.empty()) {
        stem = "gaussian_blur_output";
    }

    const char *extension = (channels == 1) ? ".pgm" : ".ppm";
    return (stem.string() + "_blurred" + extension);
}

void write_output_image(const std::string &output_image_path,
                        const float *output,
                        int image_width,
                        int image_height,
                        int channels) {
    if (output_image_path.empty()) {
        return;
    }
    if (output == nullptr) {
        throw std::runtime_error("Cannot write output image because output is null");
    }
    if (image_width <= 0 || image_height <= 0 || channels <= 0) {
        throw std::runtime_error("Cannot write output image with invalid dimensions");
    }

    std::filesystem::path output_path(output_image_path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream file(output_image_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open output image: " + output_image_path);
    }

    const auto clamp_to_u8 = [](float value) -> unsigned char {
        value = std::clamp(value, 0.0f, 1.0f);
        return static_cast<unsigned char>(value * 255.0f + 0.5f);
    };

    const std::size_t plane_pixels = static_cast<std::size_t>(image_width) * static_cast<std::size_t>(image_height);

    if (channels == 1) {
        file << "P5\n" << image_width << ' ' << image_height << "\n255\n";
        std::vector<unsigned char> row(static_cast<std::size_t>(image_width));
        for (int row_idx = 0; row_idx < image_height; ++row_idx) {
            const std::size_t row_base = static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(image_width);
            for (int column_idx = 0; column_idx < image_width; ++column_idx) {
                row[static_cast<std::size_t>(column_idx)] = clamp_to_u8(output[row_base + static_cast<std::size_t>(column_idx)]);
            }
            file.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size()));
        }
    } else if (channels >= 3) {
        file << "P6\n" << image_width << ' ' << image_height << "\n255\n";
        std::vector<unsigned char> row(static_cast<std::size_t>(image_width) * 3);
        const float *r_plane = output;
        const float *g_plane = output + plane_pixels;
        const float *b_plane = output + 2 * plane_pixels;

        for (int row_idx = 0; row_idx < image_height; ++row_idx) {
            const std::size_t row_base = static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(image_width);
            for (int column_idx = 0; column_idx < image_width; ++column_idx) {
                const std::size_t pixel_idx = row_base + static_cast<std::size_t>(column_idx);
                const std::size_t out_idx = static_cast<std::size_t>(column_idx) * 3;
                row[out_idx + 0] = clamp_to_u8(r_plane[pixel_idx]);
                row[out_idx + 1] = clamp_to_u8(g_plane[pixel_idx]);
                row[out_idx + 2] = clamp_to_u8(b_plane[pixel_idx]);
            }
            file.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size()));
        }
    } else {
        throw std::runtime_error("Unsupported channel count for output image writing");
    }

    if (!file) {
        throw std::runtime_error("Failed while writing output image: " + output_image_path);
    }
}
