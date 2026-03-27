//
// Created by aidan on 19/03/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <cstdio>
#include <limits>
#include <algorithm>

#include "cluster.h"

int main(int argc, char const *argv[]) {
    std::size_t array_length = std::stoull(argv[1]);
    std::size_t thread_count = std::stoull(argv[2]);

    std::size_t stride =
        std::ceil((float) (std::hardware_destructive_interference_size + sizeof(float)) / sizeof(float));

    float *array = new float[array_length];
    float *centre = new float[thread_count];
    float *dists = new float[stride * thread_count];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(0,std::numeric_limits<float>::epsilon() + array_length);

    for (std::size_t idx = 0; idx < array_length; ++idx) {
        array[idx] = distr(generator);
    }

    std::sort(array, array + array_length);

    std::fill_n(dists, thread_count * stride, 0.0);

    for (std::size_t idx = 0; idx < thread_count; ++idx) {
        centre[idx] = ((2.0f * (idx + 1) - 1) * array_length) / (2.0f * thread_count);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;
    double total_duration_msec = 0;

    for (int idx = 0; idx < 10; ++idx) {
        std::fill_n(dists, thread_count * stride, 0.0);
        start = std::chrono::high_resolution_clock::now();
        cluster(array_length, thread_count, array, centre, dists);
        end = std::chrono::high_resolution_clock::now();
        duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        total_duration_msec += duration_msec.count();
    }

    float *max_pointer = std::max_element(dists, dists + (thread_count * stride));

    printf("%f\n", *max_pointer);
    printf("%lu\n", (max_pointer - dists) / stride);
    printf("%f\n", (total_duration_msec / 10.0f));


    delete [] array;
    delete [] centre;
    delete [] dists;
}
