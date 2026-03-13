//
// Created by aidan on 12/03/2026.
//

#include <iostream>
#include <random>
#include <string>

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

// ACKNOWLEDGEMENT: 5% of the work is my own, 5% of the work is ChatGPT's, and 90% of the work is Professor Dan Negrut's.
// For this task, I asked ChatGPT "Is my code correct?" and provided it with my code.
// It picked up slip-ups in types used, and also spotted I forgot to change num_items to array_length somewhere.
// Otherwise, it was satisfied with my code.

using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char* argv[]) {
    int array_length = std::stoi(argv[1]);

    float h_in[array_length];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1, 1 + std::numeric_limits<float>::epsilon());

    for (size_t idx = 0; idx < array_length; ++idx) {
        h_in[idx] = distr(generator);
    }

    float* d_in = NULL;
    g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * array_length);
    cudaMemcpy(d_in, h_in, sizeof(float) * array_length, cudaMemcpyHostToDevice);

    float* d_sum = NULL;
    g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, array_length);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    float ms;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaDeviceSynchronize();

    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, array_length);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    std::cout << gpu_sum << std::endl;
    std::cout << ms  << std::endl;

    if (d_in) g_allocator.DeviceFree(d_in);
    if (d_sum) g_allocator.DeviceFree(d_sum);
    if (d_temp_storage) g_allocator.DeviceFree(d_temp_storage);
    
    return 0;
}
