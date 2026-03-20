//
// Created by aidan on 19/03/2026.
//

#include "msort.h"

#include <algorithm>

void merge(int* arr, std::size_t low, std::size_t high, int* tmp, std::size_t midpoint) {
    std::size_t idx1 = low;
    std::size_t idx2 = midpoint + 1;
    std::size_t idx = low;

    while (idx1 <= midpoint && idx2 <= high) {
        if (arr[idx1] <= arr[idx2]) {
            tmp[idx++] = arr[idx1++];
        } else {
            tmp[idx++] = arr[idx2++];
        }
    }

    while (idx1 <= midpoint) {
        tmp[idx++] = arr[idx1++];
    }

    while (idx2 <= high) {
        tmp[idx++] = arr[idx2++];
    }

    for (idx = low; idx <= high; ++idx) {
        arr[idx] = tmp[idx];
    }
}

void parallel_msort(int* arr, std::size_t low, std::size_t high, int* tmp, const std::size_t threshold) {
    if (low >= high) {
        return;
    }

    std::size_t length = high - low + 1;

    if (length < threshold) {
        std::sort(arr + low, arr + high + 1);
    } else {
        std::size_t midpoint = low +  (high - low) / 2;

#pragma omp task
        parallel_msort(arr, low, midpoint, tmp, threshold);

#pragma omp task
        parallel_msort(arr, midpoint + 1, high, tmp, threshold);

#pragma omp taskwait
        merge(arr, low, high, tmp, midpoint);
    }

}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    int tmp[n];
#pragma omp parallel
    {
#pragma omp single
        parallel_msort(arr, 0, n - 1, tmp, threshold);
    }
}