//
// Created by aidan on 12/03/2026.
//

#include <iostream>
#include <omp.h>

int main(int argc, char* argv[]) {
    omp_set_num_threads(4);

    std::cout << "Number of threads: 4" << std::endl;

#pragma omp parallel
    {
        int threadIdx = omp_get_thread_num();
#pragma omp critical // Stupid, but ensures that things print fine
        std::cout << "I am thread No. " << threadIdx << std::endl;

#pragma omp barrier // Forces relative order between thread number printing, and factorial printing

        int factorial = 1;

        for (int idx = 1; idx <= (threadIdx*2 + 1); idx++) {
            factorial *= idx;
        }

        // For some reason, these lines print fine without the need for a critical section...
        std::cout << (threadIdx*2 + 1) << "!=" << factorial << std::endl;
        std::cout << (threadIdx*2 + 2) << "!=" << (factorial*(threadIdx*2 + 2)) << std::endl;
    }

}