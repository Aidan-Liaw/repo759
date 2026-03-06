//
// Created by aidan on 5/03/2026.
//

#include <iostream>
#include <random>
#include <string>

#include "mmul.cuh"

int main(int argc, char *argv[]) {
    int array_length = std::stoi(argv[1]);
    int n_tests = std::stoi(argv[2]);

    float *mA, *mB, *mC;

    cudaMallocManaged(&mA, sizeof(float) * array_length * array_length);
    cudaMallocManaged(&mB, sizeof(float) * array_length * array_length);
    cudaMallocManaged(&mC, sizeof(float) * array_length * array_length);

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1, 1 + std::numeric_limits<float>::epsilon());

    for (size_t column_idx = 0; column_idx < array_length; ++column_idx) {
        for (size_t row_idx = 0; row_idx < array_length; ++row_idx) {
            mA[row_idx * array_length + column_idx] = distr(generator);
            mB[row_idx * array_length + column_idx] = distr(generator);
            mC[row_idx * array_length + column_idx] = 0;
        }
    }

    float ms_total = 0, ms;

    cublasHandle_t cublas;

    cublasCreate_v2(&cublas);

    for (size_t idx = 0; idx < n_tests; ++idx) {
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaDeviceSynchronize();

        mmul(cublas, mA, mB, mC, array_length);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Get the elapsed time in milliseconds
        cudaEventElapsedTime(&ms, start, stop);
        cudaDeviceSynchronize();

        ms_total += ms;
    }

    std::cout << (ms_total / (float) n_tests)  << std::endl;

    cudaFree(mA);
    cudaFree(mB);
    cudaFree(mC);

    cublasDestroy_v2(cublas);
}
