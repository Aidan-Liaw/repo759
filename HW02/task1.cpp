//
// Created by aidan on 3/02/2026.
//

// REF: https://stackoverflow.com/questions
// /9471604/what-is-the-best-way-to-generate-random-numbers-in-c
// REF: https://learn.microsoft.com/en-us/cpp
// /standard-library/uniform-real-distribution-class?view=msvc-170

#include <random>
#include <string>
#include <chrono>
#include <ratio>

#include "scan.h"

int main(int argc, char const *argv[]) {
    int N = std::stoi(argv[1]);
    float *randoms = new float[N];
    float *output = new float[N];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1,std::numeric_limits<float>::epsilon() + 1);

    for (int idx = 0; idx < N; ++idx) {
        randoms[idx] = distr(generator);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    start = std::chrono::high_resolution_clock::now();
    scan(randoms, output, N);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    printf("%.2f\n%.2f\n%.2f\n", output[0], output[N-1], duration_sec.count());

    return 0;
}
