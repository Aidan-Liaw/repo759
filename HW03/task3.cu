#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include <string>

#include "HW03/vscale.cuh"

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPT's
// ChatGPT was provided the code (as well as the vscale.cu and vscale.cuh files)
// and was asked "Is the code correct?"
// It found a bug in my number_of_blocks calculation, and found that I had
// forgotten to free the dB CUDA device array

// REF: https://stackoverflow.com/questions
// /9471604/what-is-the-best-way-to-generate-random-numbers-in-c
// REF: https://learn.microsoft.com/en-us/cpp
// /standard-library/uniform-real-distribution-class?view=msvc-170

constexpr int number_of_threads = 16;

int main(int argc, char *argv[]) {
  int array_length = std::stoi(argv[1]);

  int number_of_blocks = (array_length + number_of_threads - 1) / number_of_threads;

  float hA[array_length], *dA, hB[array_length], *dB;

  std::random_device rng_device;
  std::mt19937 generator(rng_device());
  // std::numeric_limits<float>::epsilon() is the FP epsilon
  // Which is added to ensure that 1 is included in the distribution
  // Since uniform_real_distribution is considered inclusive-exclusive
  std::uniform_real_distribution<float> distr_a(-10,10 + std::numeric_limits<float>::epsilon());
  std::uniform_real_distribution<float> distr_b(0,1 + std::numeric_limits<float>::epsilon());

  for (int idx = 0; idx < array_length; ++idx) {
    hA[idx] = distr_a(generator);
    hB[idx] = distr_b(generator);
  }

  cudaMalloc((void**)&dA, sizeof(float) * array_length);
  cudaMalloc((void**)&dB, sizeof(float) * array_length);

  cudaMemcpy(dA, hA, sizeof(float) * array_length, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(float) * array_length, cudaMemcpyHostToDevice);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  vscale<<<number_of_blocks, number_of_threads>>>(dA, dB, array_length);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get the elapsed time in milliseconds
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(hB, dB, sizeof(float) * array_length, cudaMemcpyDeviceToHost);

  std::cout << ms << std::endl;
  std::cout << hB[0] << std::endl;
  std::cout << hB[array_length - 1] << std::endl;

  cudaFree(dA);
  cudaFree(dB);

  return 0;
}
