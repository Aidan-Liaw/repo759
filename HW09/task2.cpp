//
// Created by aidan on 19/03/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <cstdio>
#include <limits>
#include <omp.h>

#include "montecarlo.h"

int main(int argc, char const *argv[]) {
    std::size_t array_length = std::stoull(argv[1]);
    std::size_t thread_count = std::stoull(argv[2]);

    omp_set_num_threads(thread_count);

    float radius = 1.0f;

    float *x_coord = new float[array_length];
    float *y_coord = new float[array_length];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-radius,std::numeric_limits<float>::epsilon() + radius);

    for (std::size_t idx = 0; idx < array_length; ++idx) {
        x_coord[idx] = distr(generator);
        y_coord[idx] = distr(generator);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;
    double total_duration_msec = 0;

    start = std::chrono::high_resolution_clock::now();
    end = std::chrono::high_resolution_clock::now();

    volatile int in_circle;
    for (int idx = 0; idx < 10; ++idx) {
        start = std::chrono::high_resolution_clock::now();
        in_circle = montecarlo(array_length, x_coord, y_coord, radius);
        end = std::chrono::high_resolution_clock::now();
        duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        total_duration_msec += duration_msec.count();
    }

    duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    float pi_estimate = (4.0f * in_circle) / (float) array_length;

    printf("%f\n", pi_estimate);
    printf("%f\n", (total_duration_msec / 10.0f));

    delete [] x_coord;
    delete [] y_coord;
}
