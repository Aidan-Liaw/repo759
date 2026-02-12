#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <random>

// ACKNOWLEDGEMENT: 95% of the work was done by me, 5% of the work was done by
// ChatGPT, by asking it "Do you think that my program is correct?" alongside
// my code. On the first pass, it found issues with initialisations and typing.
// These have been fixed.

// REF: https://stackoverflow.com/questions
// /9471604/what-is-the-best-way-to-generate-random-numbers-in-c
// REF: https://learn.microsoft.com/en-us/cpp
// /standard-library/uniform-real-distribution-class?view=msvc-170

constexpr int number_of_blocks = 2;

constexpr int number_of_threads = 8;

constexpr int array_length = number_of_blocks * number_of_threads;


// REF: https://stackoverflow.com/questions
// /11171648/factorial-method-recursive-or-iterative-java
__global__ void linear_calculation(int a, int* dA) {
  *(dA + (threadIdx.x << 1 | blockIdx.x)) = a*threadIdx.x + blockIdx.x;
}

int main() {
  int hA[array_length], *dA;

  cudaMalloc((void**)&dA, sizeof(int) * array_length);
  cudaMemset(dA, 0, sizeof(int) * array_length);

  std::random_device rng_device;
  std::mt19937 generator(rng_device());
  // std::numeric_limits<float>::epsilon() is the FP epsilon
  // Which is added to ensure that 1 is included in the distribution
  // Since uniform_real_distribution is considered inclusive-exclusive
  std::uniform_int_distribution<int> distr(-100,101);

  linear_calculation<<<number_of_blocks,number_of_threads>>>(distr(generator), dA);

  cudaMemcpy(hA, dA, sizeof(int) * array_length, cudaMemcpyDeviceToHost);

  for (int value : hA) {
    std::cout << value << std::endl;
  }

  cudaFree(dA);

  return 0;
}
