//
// Created by aidan on 3/02/2026.
//

// REF: https://stackoverflow.com/questions
// /9471604/what-is-the-best-way-to-generate-random-numbers-in-c
// REF: https://learn.microsoft.com/en-us/cpp
// /standard-library/uniform-real-distribution-class?view=msvc-170

// ACKNOWLEDGEMENT: The problem was solved by me 80% and ChatGPT 20%
// The AI model was asked to check for bugs, and found out that some variables
// were improperly defined with wrong sizes, which fixed a bug with n = 2^30
// inputs failing.

#include <random>
#include <string>
#include <chrono>
#include <ratio>

#include "scan.h"

int main(int argc, char const *argv[]) {
    std::size_t N = std::stoull(argv[1]);
    auto *randoms = new float[N];
    auto *output = new float[N];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());
    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(-1,std::numeric_limits<float>::epsilon() + 1);

    for (std::size_t idx = 0; idx < N; ++idx) {
        randoms[idx] = distr(generator);
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    start = std::chrono::high_resolution_clock::now();
    scan(randoms, output, N);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration_sec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    printf("%f\n%f\n%f\n", duration_sec.count(), output[0], output[N-1]);

    delete [] randoms;
    delete [] output;

    return 0;
}
