//
// Created by aidan on 4/02/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <cstdio>
#include <limits>

#include "convolution.h"

// The problem was solved 100% by me and 0% by ChatGPT
// However, this is because the problems that ChatGPT saw in my task1.cpp code
// were not repeated here...

int main(int argc, char const *argv[]) {
    std::size_t N = std::stoull(argv[1]);
    std::size_t M = std::stoull(argv[2]);
    float *image = new float[N*N];
    float *mask = new float[M*M];
    float *output =  new float[N*N];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> image_distr(-10,std::numeric_limits<float>::epsilon() + 10);
    std::uniform_real_distribution<float> mask_distr(-1,std::numeric_limits<float>::epsilon() + 1);

    for (std::size_t idx = 0; idx < N*N; ++idx) {
        image[idx] = image_distr(generator);
    }

    for (std::size_t idx = 0; idx < M*M; ++idx) {
        mask[idx] = mask_distr(generator);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    start = std::chrono::high_resolution_clock::now();
    convolve(image, output, N, mask, M);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    printf("%f\n%f\n%f\n", duration_sec.count(), output[0], output[N*N-1]);

    delete [] image;
    delete [] mask;
    delete [] output;

    return 0;
}
