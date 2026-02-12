#include <iostream>
#include <cuda.h>
#include <stdio.h>


// ACKNOWLEDGEMENT: No AIs/LLMs were used.
// Code is based off sample code from slides.

constexpr int number_of_blocks = 1;

constexpr int number_of_threads = 8;


// REF: https://stackoverflow.com/questions
// /11171648/factorial-method-recursive-or-iterative-java
__global__ void factorial(int* dA) {
  int total = 1;
  for (int idx = 1; idx <= threadIdx.x + 1; ++idx) {
    total = total * idx;
  }
  *(dA + threadIdx.x) = total;
}

int main() {
  int hA[number_of_threads], *dA;

  cudaMalloc((void**)&dA, sizeof(int) * number_of_threads);
  cudaMemset(dA, 0, sizeof(int) * number_of_threads);

  factorial<<<number_of_blocks,number_of_threads>>>(dA);

  cudaMemcpy(hA, dA, sizeof(int) * number_of_threads, cudaMemcpyDeviceToHost);

  for (int factorial_value : hA) {
    std::cout << factorial_value << std::endl;
  }

  cudaFree(dA);

  return 0;
}
