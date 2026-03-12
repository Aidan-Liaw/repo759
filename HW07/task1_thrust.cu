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
    thrust::device_vector<int> dV = hV;

    float ms;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaDeviceSynchronize();

    float sum = thrust::reduce(dV.begin(), dV.end(), 0, thrust::plus<float>());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    std::cout << ms  << std::endl;
    std::cout << sum << std::endl;

    return 0;
}
