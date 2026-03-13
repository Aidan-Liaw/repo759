//
// Created by aidan on 12/03/2026.
//

#include <iostream>
#include <random>
#include <string>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// ACKNOWLEDGEMENT: 90% of the work is my own, and 10% of the work is ChatGPT's.
// For this task, I asked ChatGPT "Is my code correct?" and provided it with my code.
// It picked up slip-ups in types used, and also spotted that thrust::reduce should use a floating constant to
// ensure that it did float operations, which Clang-Tidy agreed with.
// Otherwise, it was satisfied with my code.


int main(int argc, char* argv[]) {
    int array_length = std::stoi(argv[1]);

    thrust::host_vector<float> hV(array_length);

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1, 1 + std::numeric_limits<float>::epsilon());

    for (size_t idx = 0; idx < array_length; ++idx) {
        hV[idx] = distr(generator);
    }

    // Technically speaking, the = sign is a built-in Thrust function
    // See its operator overload...
    thrust::device_vector<float> dV = hV;

    float ms;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaDeviceSynchronize();

    float sum = thrust::reduce(dV.begin(), dV.end(), 0.0f, thrust::plus<float>());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    std::cout << sum << std::endl;
    std::cout << ms  << std::endl;

    return 0;
}
