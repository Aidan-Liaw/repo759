//
// Created by aidan on 19/02/2026.
//

// Please see task2.cu for comments on AI usage.

#include "stencil.cuh"

__global__ void stencil_kernel(const float *image, const float *mask, float *output, unsigned int n, unsigned int R) {
    int64_t thread_idx = (int64_t) blockIdx.x * (int64_t) blockDim.x + (int64_t) threadIdx.x;
    int64_t block_offset = (int64_t) blockIdx.x * (int64_t) blockDim.x - (int64_t) R;

    extern __shared__ float shared_memory[];
    float *sImage = shared_memory;
    float *sMask = sImage + (2 * R + blockDim.x);
    float *sOutput = sMask + (2 * R + 1);

    for (int idx = threadIdx.x; idx < 2 * R + blockDim.x; idx += blockDim.x) {
        if (block_offset + idx >= 0 && block_offset + idx < n) {
            *(sImage + idx) = *(image + block_offset + idx);
        }
    }

    for (int idx = threadIdx.x; idx < 2 * R + 1; idx += blockDim.x) {
        *(sMask + idx) = *(mask + idx);
    }

    // Synchronise to ensure that the matrices are loaded
    __syncthreads();

    if (thread_idx >= (int64_t) n) {
        return;
    }

    *(sOutput + threadIdx.x) = 0;
    for (int64_t j = -((int64_t) R); j <= R; ++j) {
        float image_element = thread_idx + j < 0 || thread_idx + j >= n
                                  ? 1
                                  : *(sImage + threadIdx.x + j + R);
        *(sOutput + threadIdx.x) += image_element * *(sMask + j + R);
    }

    *(output + thread_idx) = *(sOutput + threadIdx.x);
}


// threads_per_block >= 2 * R + 1
__host__ void stencil(const float *image,
                      const float *mask,
                      float *output,
                      unsigned int n,
                      unsigned int R,
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

    stencil_kernel<<<number_of_blocks, threads_per_block, shared_memory_size>>>(image, mask, output, n, R);
}
