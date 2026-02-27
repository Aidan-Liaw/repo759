//
// Created by aidan on 26/02/2026.
//

// ACKNOWLEDGEMENT: 50% of the work is my own, 30% of the work is ChatGPT's, and 20% of the work is the lecture slides
// For this task, I did not ask ChatGPT "Is the code correct?", at the start, as it kept giving me solutions
// (that were wrong because it lacked the full context, or were right and reduced my ability to claim ownership over the code).
// So instead it was asked "Is where I'm at good?" to ensure that it was prompted to only provide feedback.
// Since I was confused on why there were two levels of indirection for the host function, I did additionally ask it:
// "Is the reason why there are two levels of indirection because it allows for the ping-ponging to take place between input and output arrays?"
// Which helped clarify what I am meant to be doing.

#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = (i >= n ? 0 : g_idata[i]) + (i + blockDim.x >= n ? 0 : g_idata[i + blockDim.x]);
    __syncthreads();

    if (tid == 0) {
        g_odata[blockIdx.x];
    } else if (i < n ) {
        g_odata[i] = sdata[tid];
    }

}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
    size_t number_of_blocks = (n + 2 * threads_per_block - 1) / (2 * threads_per_block);

    for (size_t idx = N; idx > 1; idx /= 2) {
        reduce_kernel<<<number_of_blocks, threads_per_block, idx * sizeof(float)>>>(*input, *output, idx);
    }
}
