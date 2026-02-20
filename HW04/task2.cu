#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include <string>

#include "stencil.cuh"

// ACKNOWLEDGEMENT: 20% of the work is my own, and 80% of the work is ChatGPT's
// ChatGPT was provided the code (as well as the stencil.cu and stencil.cuh files)
// and was asked "Is the code correct?" multiple times.
// Everything was wrong. I did not respect the 1D kernel (again).
// I approached the problem with a 2D configuration as I had misunderstood the task as the arrays being 2D.
// I also misunderstood the requirements of what should be in shared memory
// (I originally dumped everything into shared memory, which ChatGPT fixed.)

// Lots of small mistakes were also present that were not related to assuming a 2D configuration.
// A majority of these issues boiled down to indexing. One example was
// (after I had done an initial fix to use a 1D configuration), not including a block_offset
// to ensure correctly access sImage. As my sImage instantiation loop starts at threadIdx.x,
// and ends at 2 * R + blockDim.x (which is indeed the correct number of elements)
// the problem was that I needed to start at blockDim.x - R and end at blockDim.x + R
// However, this does not apply to sImage it only holds the minimum number of required elements
// to perform the calculation, so it does not need the offset.

// There was also some confusion with how thread_idx works. I had used thread_idx for the accessing the elements
// needed during the computation as I was blindly implementing the algorithm (and the variables used).
// ChatGPT corrected me to using threadIdx.x for any shared memory access.

// Most of these failures boil down to using the wrong indexing system for shared memory,
// versus global memory.

// Overall, the intent was correct, but the assumptions and execution was wrong.
// Even after fixing the 2D to 1D conversion (without ChatGPT's help as
// one prompt did ask whether the global memory approach was correct, as I did that before trying shared memory)
// the wrong indexing was a major issue.
// As a result, I only claim 20% of my work to be my own. Upon reflection, the code was majorly problematic.

// ChatGPT also made some comments on compilation (like using VLAs), but ignored them
// because the code compiles and works. You may also see some weird type casting
// (even though I am well aware that C++ will promote types they way my explicit type casting does).
// I did it mostly so that when I asked whether my code was correct, ChatGPT would not complain.


// REF: https://stackoverflow.com/questions
// /9471604/what-is-the-best-way-to-generate-random-numbers-in-c
// REF: https://learn.microsoft.com/en-us/cpp
// /standard-library/uniform-real-distribution-class?view=msvc-170

int main(int argc, char *argv[]) {
    int image_length = std::stoi(argv[1]);
    int R = std::stoi(argv[2]);
    int mask_length = 2 * R + 1;
    int threads_per_block = std::stoi(argv[3]);

    float image[image_length];
    float mask[mask_length];
    float output[image_length];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1, 1 + std::numeric_limits<float>::epsilon());

    for (size_t idx = 0; idx < image_length; ++idx) {
        image[idx] = distr(generator);
    }

    for (size_t idx = 0; idx < mask_length; ++idx) {
        mask[idx] = distr(generator);
    }


    float *gImage, *gMask, *gOutput;

    cudaMalloc((void **) &gImage, sizeof(image));
    cudaMalloc((void **) &gMask, sizeof(mask));
    cudaMalloc((void **) &gOutput, sizeof(output));

    cudaMemcpy(gImage, image, sizeof(image), cudaMemcpyHostToDevice);
    cudaMemcpy(gMask, mask, sizeof(mask), cudaMemcpyHostToDevice);

#if PERF_TEST == 1
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
#else
    cudaDeviceSynchronize();
#endif

    stencil(gImage, gMask, gOutput, image_length, R, threads_per_block);

#if PERF_TEST == 1
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
#else
    cudaDeviceSynchronize();
#endif

    cudaMemcpy(output, gOutput, sizeof(output), cudaMemcpyDeviceToHost);

#if PERF_TEST == 1
  std::cout << ms << std::endl;
  std::cout << output[image_length - 1] << std::endl;
#endif

    cudaFree(gImage);
    cudaFree(gMask);
    cudaFree(gOutput);

    return 0;
}
