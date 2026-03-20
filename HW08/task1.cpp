//
// Created by aidan on 19/03/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <cstdio>
#include <limits>
#include <omp.h>

#include "matmul.h"

int main(int argc, char const *argv[]) {
    std::size_t arrayLength = std::stoull(argv[1]);
    std::size_t thread_count = std::stoull(argv[2]);

    omp_set_num_threads(thread_count);

    float *A = new float[arrayLength*arrayLength];
    float *B = new float[arrayLength*arrayLength];
    float *C = new float[arrayLength*arrayLength];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<double> AB_distr(-10,std::numeric_limits<double>::epsilon() + 10);

    for (std::size_t idx = 0; idx < arrayLength*arrayLength; ++idx) {
        A[idx] = AB_distr(generator);
        B[idx] = AB_distr(generator);
    }

    std::fill_n(C, arrayLength * arrayLength, 0.0);

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;

    start = std::chrono::high_resolution_clock::now();
    mmul(A, B, C, arrayLength);
    end = std::chrono::high_resolution_clock::now();
    duration_msec =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    printf("%f\n", C[0]);
    printf("%f\n", C[arrayLength - 1]);
    printf("%f\n", duration_msec.count());


    delete [] A;
    delete [] B;
    delete [] C;
}
