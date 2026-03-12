//
// Created by aidan on 12/03/2026.
//

#include <omp.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    omp_set_num_threads(4);

    std::cout << "Number of threads: 4" << std::endl;
#pragma omp parallel
{
    int threadIdx = omp_get_thread_num();
    int factorial = 1;

    for (int idx = 1; idx <= (threadIdx*2 + 1); idx++) {
        factorial *= idx;
    }

    std::cout << "Number of threads: " << threadIdx << std::endl;
    std::cout << (threadIdx*2 + 1) << "!=" << factorial << std::endl;
    std::cout << (threadIdx*2 + 2) << "!=" << (factorial*(threadIdx*2 + 2)) << std::endl;
}