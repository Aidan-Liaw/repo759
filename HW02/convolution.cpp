//
// Created by aidan on 3/02/2026.
//

#include "convolution.h"

// REF: This problem was solved by me 30% and ChatGPT 70%. 
// Code is heavily inspired by ChatGPTs responses. Whatever I did not understand
// from its response is omitted from my answer.
// See conversation in link below:
// https://chatgpt.com/share/69839a4f-c040-8005-acb2-acb4c1c5c616

static inline float get_image_element(const float *image, std::size_t n, long long output_idx1, long long output_idx2) {
    bool is_output_idx1_valid = output_idx1 < n && output_idx1 >= 0;
    bool is_output_idx2_valid = output_idx2 < n && output_idx2 >= 0;


    if (is_output_idx1_valid && is_output_idx2_valid) {
        return image[output_idx1 * n + output_idx2];
    }

    // Both are outside bounds, so corner
    if (!is_output_idx1_valid && !is_output_idx2_valid) {
        return 0;
    }

    return 1;
}

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    // Guaranteed to be an integer, as m is an odd number
    std::size_t k = (m - 1) / 2;
    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float sum = 0;
            for (std::size_t i = 0; i < m; i++) {
		std::size_t output_idx1 = static_cast<std::size_t>(x + i) - k;
                for (std::size_t j = 0; j < m; j++) {
                    std::size_t output_idx2 = static_cast<std::size_t >(y + j) - k;
                    float modifier = get_image_element(image, n, output_idx1, output_idx2);
                    sum += mask[i * m + j] * modifier;
                }
            }
            output[x * n + y] = sum;
        }
    }
}
