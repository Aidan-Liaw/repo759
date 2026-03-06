//
// Created by aidan on 5/03/2026.
//

#include "scan.cuh"

// ACKNOWLEDGEMENT: 40% of the work is my own, annd 60% of the work is ChatGPT's
// ChatGPT was provided with the following prompt multiple times:

// Is my code correct? 
// Note that both input and output use managed memory, meaning that the GPU should have access to this memory. I am aware of the performance bottlenecks, but that is not of my concern. 
// Assume that I will run this multiple times, and that only 1, 1D block will be running at a time. I will perform the inclusive scan in chunks, as it is permitted by my assignment.
// There cannot be any processing done in the host function. You should not make major changes to the code to fix it, only line by line edits.// 
// Expects only 1 block of threads to run, and that 1 block should be 1D.

// Earlier prompts simply asked ChatGPT "Is my code correct?", and it made assumptions that were wrong. What you see above is the final prompt that I used in the end. For example, it suggested that I use an array that is double the size of the input array (which would no doubt work and also score me very little marks).

// As usual, it picked up small issues, like mistakes I made when I (both blindly and incorrectly) copied code off the lecture slides (such as writing p_in instead of p_out in certain places), but also more critical mistakes.
// It however picked up a more critical mistake, that blocks are not sychronised, and that it would be a fool's errand to try to do so. I also tried to access shared memory by assuming it was globally accessible, instead of accessible only on a per-block basis.

// After looking through Piazza, I instead took the approach of compelting the inclusive scan over multiple kernel calls.
// It picked up more errors after like using the wrong upper bound for my for loop, or forgetting to add the last array value of the previous call (or 0 if it was the first call).

// After these fixes, the code was told that it would work by ChatGPT (but it never said how slow it would be...).

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
