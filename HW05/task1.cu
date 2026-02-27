#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include <string>

#include "matmul.cuh"

// ACKNOWLEDGEMENT: 30% of the work is my own, and 70% of the work is ChatGPT's
// ChatGPT was provided the code (as well as the matmul.cu and matmul.cuh files)
// and was asked "Is the code correct?" multiple times.
// Everything was wrong. I did not respect the 1D kernel. My port of the loop from HW02 was wrong.
// I had failed to understand that the array is indeed 2D.

// As a result, most of the code (except the for loop, array indexing
// and the thread_idx which I took off the slides) is ChatGPT's

// The only "bug" I found was that my device memory allocation was in the wrong place. It should be
// done here, not in the matmul.cu file's functions.

// Note that the plot for this task looks stupid. Very stupid, but multiple runs provide the same result.
// I cannot find the bug in my code that causes this. Apologies.

// REF: https://stackoverflow.com/questions
// /9471604/what-is-the-best-way-to-generate-random-numbers-in-c
// REF: https://learn.microsoft.com/en-us/cpp
// /standard-library/uniform-real-distribution-class?view=msvc-170

int main(int argc, char *argv[]) {
    int array_length = std::stoi(argv[1]);
    int block_dimension = std::stoi(argv[2]);

    float hA[array_length * array_length], hB[array_length * array_length], hC[array_length * array_length];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1, 1 + std::numeric_limits<float>::epsilon());

    for (int idx = 0; idx < array_length * array_length; ++idx) {
        hA[idx] = distr(generator);
        hB[idx] = distr(generator);
    }


    float *dA, *dB, *dC;

    cudaMalloc((void **) &dA, sizeof(float) * array_length * array_length);
    cudaMalloc((void **) &dB, sizeof(float) * array_length * array_length);
    cudaMalloc((void **) &dC, sizeof(float) * array_length * array_length);

    for (int idx = 0; idx < 3; ++idx) {
        cudaMemcpy(dA, hA, sizeof(float) * array_length * array_length, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, sizeof(float) * array_length * array_length, cudaMemcpyHostToDevice);

#if PERF_TEST == 1
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
#else
        cudaDeviceSynchronize();
#endif
        switch (idx) {
            case 0:
                matmul_1(dA, dB, dC, array_length, block_dimension);
                break;
            case 1:
                matmul_2(dA, dB, dC, array_length, block_dimension);
                break;
            case 2:
                matmul_3(dA, dB, dC, array_length, block_dimension);
                break;
            default:
                break;
        }

#if PERF_TEST == 1
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Get the elapsed time in milliseconds
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
#else
        cudaDeviceSynchronize();
#endif

        cudaMemcpy(hC, dC, sizeof(float) * array_length * array_length, cudaMemcpyDeviceToHost);

#if PERF_TEST == 1
        std::cout << ms << std::endl;
        std::cout << hC[array_length * array_length - 1] << std::endl;
#endif
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}