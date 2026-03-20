//
// Created by aidan on 19/03/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <cstdio>
#include <limits>
#include <omp.h>

#include "convolution.h"

int main(int argc, char const *argv[]) {
    std::size_t arrayLength = std::stoull(argv[1]);
    std::size_t thread_count = std::stoull(argv[2]);

    omp_set_num_threads(thread_count);
    float *image = new float[arrayLength*arrayLength];
    float *mask = new float[3*3];
    float *output =  new float[arrayLength*arrayLength];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> image_distr(-10,std::numeric_limits<float>::epsilon() + 10);
    std::uniform_real_distribution<float> mask_distr(-1,std::numeric_limits<float>::epsilon() + 1);

    for (std::size_t idx = 0; idx < arrayLength*arrayLength; ++idx) {
        image[idx] = image_distr(generator);
    }

    for (std::size_t idx = 0; idx < 3*3; ++idx) {
        mask[idx] = mask_distr(generator);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    start = std::chrono::high_resolution_clock::now();
    convolve(image, output, arrayLength, mask, 3);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    printf("%f\n%f\n%f\n", output[0], output[arrayLength*arrayLength-1], duration_msec.count());

    delete [] image;
    delete [] mask;
    delete [] output;
}
