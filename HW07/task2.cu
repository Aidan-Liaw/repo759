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

#include "count.cuh"


int main(int argc, char* argv[]) {
    int array_length = std::stoi(argv[1]);

    thrust::host_vector<int> hV(array_length);

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_int_distribution<int> distr(0, 501);

    for (size_t idx = 0; idx < array_length; ++idx) {
        hV[idx] = distr(generator);
    }

    // Technically speaking, the = sign is a built-in Thrust function
    // See its operator overload...
    thrust::device_vector<int> dV = hV;
    thrust::device_vector<int> dValues(array_length);
    thrust::device_vector<int> dCount(array_length);

    float ms;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaDeviceSynchronize();

    count(dV, dValues, dCount);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    std::cout << dValues.back() << std::endl;
    std::cout << dCount.back() << std::endl;
    std::cout << ms  << std::endl;

    return 0;
}
