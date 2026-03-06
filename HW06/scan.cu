//
// Created by aidan on 5/03/2026.
//

#include "scan.cuh"

// Expects only 1 block of threads to run, and that 1 block should be 1D.
__global__ void hillis_steele(const float* input, float* output, unsigned int n, unsigned int input_output_offset) {
    extern volatile __shared__ float temp[];
    float value;

    int idx = threadIdx.x;
    float carry = input_output_offset != 0 ? output[input_output_offset - 1] : 0;
    int upper_bound = n - input_output_offset < blockDim.x ? n - input_output_offset : blockDim.x;

    if (idx < upper_bound) {
        temp[idx] = input[input_output_offset + idx];
    }

    __syncthreads();

    for (int offset = 1; offset < upper_bound; offset *= 2) {
        if (idx < upper_bound) {
            if (idx >= offset) {
                value = temp[idx] + temp[idx - offset];
            } else {
                value = temp[idx];
            }
        }
        __syncthreads();

        if (idx < upper_bound) {
            temp[idx] = value;
        }

        __syncthreads();
    }

    if (idx < upper_bound) {
        output[input_output_offset + idx] = carry + temp[idx];
    }
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    for (size_t offset = 0; offset < n; offset += threads_per_block) {
        hillis_steele<<<1, threads_per_block,  threads_per_block * sizeof(float)>>>(input, output, n, offset);

        cudaDeviceSynchronize();
    }
}
