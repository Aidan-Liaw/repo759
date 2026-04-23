#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// REF: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html

__global__ void busy_wait_kernel(volatile long long endTime) {
    if (threadIdx.x == 0) {
        volatile long long currentTime = clock64();
        while (endTime > currentTime) {
            currentTime = clock64();
        }
    }
    __syncthreads();
}

__global__ void critical_kernel(volatile long long *enterTime) {
    if (threadIdx.x == 0) {
        *enterTime = clock64();
    }
    __syncthreads();
}

int main() {
    int gpu_device_index = 0; // GPU ordinal
    (cudaSetDevice(gpu_device_index));

    /* ------------------ Code required to create green contexts --------------------------- */

    // Get all available GPU SM resources
    CUdevResource initial_GPU_SM_resources{};
    cuDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, CU_DEV_RESOURCE_TYPE_SM);

    // Get the number of possible groups that can be created, each group haing 8 SMs
    unsigned int nbGroups;
    CUdevResource remainder_res{};
    cuDevSmResourceSplitByCount(nullptr, &nbGroups, &initial_GPU_SM_resources,nullptr,0,8);

    // Split SM resources
    std::vector<CUdevResource> groups(nbGroups);

    cuDevSmResourceSplitByCount(groups.data(), &nbGroups, &initial_GPU_SM_resources, &remainder_res, 0, 8);

    // Generate resource descriptor
    CUdevResourceDesc critical_desc{};
    cuDevResourceGenerateDesc(&critical_desc, &groups[0], 1);

    std::vector<CUdevResource> main_resources;
    for (std::size_t idx = 1; idx < nbGroups; ++idx) {
        main_resources.push_back(groups[idx]);
    }

    if (remainder_res.type == CU_DEV_RESOURCE_TYPE_SM && remainder_res.sm.smCount > 0) {
        main_resources.push_back(remainder_res);
    }

    CUdevResourceDesc main_desc{};
    cuDevResourceGenerateDesc(&main_desc, main_resources.data(), main_resources.size());

    // Create green context
    CUgreenCtx critical_gc{};
    cuGreenCtxCreate(&critical_gc, critical_desc, gpu_device_index, CU_GREEN_CTX_DEFAULT_STREAM );
    CUgreenCtx main_gc{};
    cuGreenCtxCreate(&main_gc, main_desc, gpu_device_index, CU_GREEN_CTX_DEFAULT_STREAM );

    // Create streams that belong to green contexts
    CUstream critical_stream{};
    cuGreenCtxStreamCreate(&critical_stream, critical_gc, CU_STREAM_NON_BLOCKING, 0);
    CUstream main_stream{};
    cuGreenCtxStreamCreate(&main_stream, main_gc, CU_STREAM_NON_BLOCKING, 0);

    /* ------------------ End GC Setup --------------------------- */

    // No need to modify any code in this function or in your kernel(s).
    // Reminder: what is abstracted in this function + kernels is the vast majority of your code
    // Now kernel(s) running on stream strm1 will use at most 16 SMs and kernel(s) on strm2 at most 8 SMs.
    long long startTime = clock64();
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, gpu_device_index);
    int clockRatekHz;
    cudaDeviceGetAttribute(&clockRatekHz, cudaDevAttrClockRate, gpu_device_index);

    long long endTime = (clockRatekHz * 1000) * 120 + startTime;
    long long criticalTime = (clockRatekHz * 1000) * (rand() % (100 - 20 + 1) + 20) + startTime;

    long long *enterTime;
    cudaMallocHost(&enterTime, sizeof(long long), cudaHostAllocMapped);


    busy_wait_kernel<<<,,0,main_stream>>>(endTime);
    while (clock64() < criticalTime) {
        // Idle until it's time
    }
    critical_kernel<<<,,0,critical_stream>>>(enterTime);

    std::cout << enterTime - criticalTime << std::endl;

    // cleanup code not shown

    return 0;
}