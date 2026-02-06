//
// Created by aidan on 4/02/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <cstdio>
#include <limits>
#include <vector>

#include "convolution.h"
#include "matmul.h"

const int arrayLength = 1000;

// REF: https://cplusplus.com/forum/beginner/27582/
//
// The problem was solved 100% by me and 0% by ChatGPT
// However, this is because the problems that ChatGPT saw in my task1.cpp code
// were not repeated here...

typedef void (*mmul_array_t)(const double*, const double*, double*, const unsigned int);

constexpr mmul_array_t mmul_array[] = {&mmul1, &mmul2, &mmul3};

typedef void (*mmul_vect_t)(const std::vector<double>&, const std::vector<double>&, double*, const unsigned int);

constexpr mmul_vect_t mmul_vect[] = {&mmul4};

int main(int argc, char const *argv[]) {
    double *A = new double[arrayLength*arrayLength];
    double *B = new double[arrayLength*arrayLength];
    double *C = new double[arrayLength*arrayLength];
    double last_value[4];


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

    std::size_t time_point_idx = 0;
    std::chrono::high_resolution_clock::time_point start[4];
    std::chrono::high_resolution_clock::time_point end[4];
    std::chrono::duration<double, std::milli> duration_sec[4];


    for (auto mmul_func: mmul_array) {
        start[time_point_idx] = std::chrono::high_resolution_clock::now();
        mmul_func(A, B, C, arrayLength);
        end[time_point_idx] = std::chrono::high_resolution_clock::now();
        last_value[time_point_idx] = C[arrayLength*arrayLength - 1];
        duration_sec[time_point_idx] =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                end[time_point_idx] - start[time_point_idx]);
        time_point_idx++;
    }

    std::vector<double> A_vec(A, A + arrayLength * arrayLength);
    std::vector<double> B_vec(B, B + arrayLength * arrayLength);
    for (auto mmul_func: mmul_vect) {
        start[time_point_idx] = std::chrono::high_resolution_clock::now();
        mmul_func(A_vec, B_vec, C, arrayLength);
        end[time_point_idx] = std::chrono::high_resolution_clock::now();
        last_value[time_point_idx] = C[arrayLength*arrayLength - 1];
        duration_sec[time_point_idx] =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                end[time_point_idx] - start[time_point_idx]);
        time_point_idx++;
    }

    printf("%d\n", arrayLength);
    for (std::size_t idx = 0; idx < time_point_idx; ++idx) {
        printf("%f\n%f\n", duration_sec[idx].count(), last_value[idx]);
    }

    delete [] A;
    delete [] B;
    delete [] C;

    return 0;
}
