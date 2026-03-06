//
// Created by aidan on 5/03/2026.
//

#include <iostream>
#include <random>
#include <string>

#include "mmul.cuh"
#include "scan.cuh"

int main(int argc, char *argv[]) {
    int array_length = std::stoi(argv[1]);
    int threads_per_block = std::stoi(argv[2]);

    float *mInput, *mOutput;

    cudaMallocManaged(&mInput, sizeof(float) * array_length);
    cudaMallocManaged(&mOutput, sizeof(float) * array_length);

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1, 1 + std::numeric_limits<float>::epsilon());

    for (size_t idx = 0; idx < array_length; ++idx) {
        mInput[idx] = distr(generator);
        mOutput[idx] = 0;
    }

    float ms;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaDeviceSynchronize();

    scan(mInput, mOutput, array_length, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    std::cout << ms  << std::endl;

    cudaFree(mInput);
    cudaFree(mOutput);
}
