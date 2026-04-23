//
// Created by aidan on 19/04/2026.
//

#include "convolution.cuh"

__global__ void horizontal_convolve_kernel(const float *image, float *output, long long dim_len, long long pitch,
    const float *mask, long long R) {
    int64_t thread_idx = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    int64_t block_offset = (int64_t) blockIdx.x * (int64_t) blockDim.x - (int64_t) R;

    int64_t row_idx = thread_idx / dim_len;
    int64_t column_idx = thread_idx % dim_len;

    extern __shared__ float shared_memory[];
    float *sImage = shared_memory;
    float *sMask = sImage + (2 * R + blockDim.x);

    // REF: https://docs.nvidia.com/cuda/cuda-runtime-api
    // /group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c

    int64_t image_idx = threadIdx.x;

    for (; (block_offset + image_idx < 0) && (image_idx < 2 * R + blockDim.x); image_idx += blockDim.x) {
        *(sImage + image_idx) = *((float*)((char*)image + row_idx * pitch) + 0);
    }

    for (; (block_offset + image_idx < dim_len) && (image_idx < 2 * R + blockDim.x); image_idx += blockDim.x) {
        *(sImage + image_idx) = *((float*)((char*)image + row_idx * pitch) + image_idx);
    }

    for (; image_idx < 2 * R + blockDim.x; image_idx += blockDim.x) {
        *(sImage + image_idx) = *((float*)((char*)image + row_idx * pitch) + (dim_len - 1));
    }


    /*for (int64_t idx = threadIdx.x; idx < 2 * R + blockDim.x; idx += blockDim.x) {
        if (block_offset + idx >= 0 && block_offset + idx < dim_len) {
            *(sImage + idx) = *(image + block_offset + idx);
        } else if (block_offset + idx < 0) {
            *(sImage + idx) = *(image);
        } else {
            *(sImage + idx) = *(image + (dim_len - 1));
        }
    }*/

    for (int64_t idx = threadIdx.x; idx < 2 * R + 1; idx += blockDim.x) {
        *(sMask + idx) = *(mask + (2 * R + 1 - 1 - idx));
    }

    // Synchronise to ensure that the matrices are loaded
    __syncthreads();

    if (row_idx >= (int64_t) dim_len) {
        return;
    }

    float sum = 0.0f;
    for (int64_t j = -((int64_t) R); j <= R; ++j) {
        sum += *(sImage + threadIdx.x + j + R) * *(sMask + j + R);
    }

    *(output + thread_idx) = sum;
}

__host__ void convolve(const float *image,
                      const float *mask,
                      float *output,
                      long long n,
                      long long R,
                      long long pitch,
                      unsigned int threads_per_block) {
    size_t number_of_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Accounts for sImage, sMask, and sOutput
    // threads_per_block accounts for how each thread needs its own output space,
    // and how sImage needs to be threads_per_block larger so that different thread indexes can access
    // the required elements of the image array.
    // Since memory is shared, there is significant overlap between used image data,
    // and so threads_per_block alongside the offset in the kernel function ensures that the lowest and highest
    // index threads cam still access everything it nees.
    size_t shared_memory_size = ((2 * R + threads_per_block) + (2 * R + 1) + threads_per_block) * sizeof(float);

    horizontal_convolve_kernel<<<number_of_blocks, threads_per_block, shared_memory_size>>>(image, output, n, mask, R);
}