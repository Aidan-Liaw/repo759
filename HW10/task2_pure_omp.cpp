//
// Created by aidan on 10/04/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <omp.h>

#include "optimize.h"
#include "reduce.h"

int main(int argc, char *argv[]) {
    std::size_t array_length = std::stoull(argv[1]);
    std::size_t thread_count = std::stoull(argv[2]);

    omp_set_num_threads(thread_count);

    float *arr = new float[array_length];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());

    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(0,std::numeric_limits<float>::epsilon() + 10);
    for (std::size_t idx = 0; idx < array_length; ++idx) {
        arr[idx] = distr(generator);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;

    start = std::chrono::high_resolution_clock::now();
    float res = reduce(arr, 0, array_length);
    end = std::chrono::high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start);

    printf("%f\n", res);
    printf("%f\n", duration_msec.count());

    delete[] arr;
}
