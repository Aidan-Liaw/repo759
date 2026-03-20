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
#include "msort.h"

int main(int argc, char const *argv[]) {
    std::size_t arrayLength = std::stoull(argv[1]);
    std::size_t thread_count = std::stoull(argv[2]);
    std::size_t threshold = std::stoull(argv[3]);

    omp_set_num_threads(thread_count);

    int *arr = new int[arrayLength];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_int_distribution<int> distr(-1000, 1000 + 1);

    for (std::size_t idx = 0; idx < arrayLength; ++idx) {
        arr[idx] = distr(generator);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;

    start = std::chrono::high_resolution_clock::now();
    msort(arr, arrayLength, threshold);
    end = std::chrono::high_resolution_clock::now();
    duration_msec =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    printf("%d\n", arr[0]);
    printf("%d\n", arr[arrayLength - 1]);
    printf("%f\n", duration_msec.count());


    delete [] arr;
}
