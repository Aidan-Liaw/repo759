#include "cluster.h"
#include <cmath>
#include <iostream>

// ACKNOWLEDGEMENT: 60% of the work is my own, and 40% of the work is ChatGPT's
// ChatGPT was provided the code and was asked "Is the code correct?".
// It found issues with omp pragma usage, and my stride calculation.
// It also pointed out that I was originally incorrectly attempting a reduction,
// which ended up with me trying a different approach

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
    // Ensures that accesses occur on a per cache line basis
    std::size_t stride =
        std::ceil((float) (std::hardware_destructive_interference_size + sizeof(float)) / sizeof(float));

    #pragma omp parallel num_threads(t)
    {
        unsigned int tid = omp_get_thread_num();
        // Local buffer gets rid of false sharing if stride is not used (or set to 1)
        float dist = 0.0f;

        #pragma omp for simd schedule(static)
        // https://stackoverflow.com/questions/36526259/split-c-array-into-n-equal-parts
        for (size_t i = 0; i < n; i++) {
            dist += std::fabs(arr[i] - centers[tid]);
        }
        dists[tid * stride] = dist;
    }
}