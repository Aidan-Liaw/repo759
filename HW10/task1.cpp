//
// Created by aidan on 10/04/2026.
//

#include <chrono>
#include <random>
#include <string>

#include "optimize.h"

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPT's
// ChatGPT was provided the code and was asked "Is the code correct?".
// It picked up __builtin_types_compatible_p is not available in C++
// and pushed me to use std::is_same_v instead

int main(int argc, char const *argv[]) {
    std::size_t vector_length = std::stoull(argv[1]);

    vec v = vec(vector_length);
    v.data = (data_t*) malloc(vector_length * sizeof(data_t));

    std::random_device rng_device;
    std::mt19937 generator(rng_device());

    if constexpr (std::is_same_v<data_t, int>) {
        std::uniform_int_distribution<int> distr(0, 11);
        for (std::size_t idx = 0; idx < vector_length; ++idx) {
            v.data[idx] = distr(generator);
        }
    } else {
        // std::numeric_limits<float>::epsilon() is the FP epsilon
        // Which is added to ensure that 1 is included in the distribution
        // Since uniform_real_distribution is considered inclusive-exclusive
        std::uniform_real_distribution<float> distr(0,std::numeric_limits<float>::epsilon() + 10);
        for (std::size_t idx = 0; idx < vector_length; ++idx) {
            v.data[idx] = distr(generator);
        }
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;

    for (int functIdx = 0; functIdx < 5; ++functIdx) {
        double total_duration_msec = 0;
        data_t dest;
        for (int idx = 0; idx < 10; ++idx) {
            dest = IDENT;
            switch (functIdx) {
                case 0:
                    start = std::chrono::high_resolution_clock::now();
                    optimize1(&v, &dest);
                    end = std::chrono::high_resolution_clock::now();
                    break;
                case 1:
                    start = std::chrono::high_resolution_clock::now();
                    optimize2(&v, &dest);
                    end = std::chrono::high_resolution_clock::now();
                    break;
                case 2:
                    start = std::chrono::high_resolution_clock::now();
                    optimize3(&v, &dest);
                    end = std::chrono::high_resolution_clock::now();
                    break;
                case 3:
                    start = std::chrono::high_resolution_clock::now();
                    optimize4(&v, &dest);
                    end = std::chrono::high_resolution_clock::now();
                    break;
                case 4:
                    start = std::chrono::high_resolution_clock::now();
                    optimize5(&v, &dest);
                    end = std::chrono::high_resolution_clock::now();
                    break;
            }
            duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start);
            total_duration_msec += duration_msec.count();
        }
        if constexpr (std::is_same_v<data_t, int>) {
            printf("%d\n", dest);
        } else {
            printf("%f\n", dest);
        }
        printf("%f\n", total_duration_msec / 10.0);
    }

    free(v.data);
}
