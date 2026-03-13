//
// Created by aidan on 12/03/2026.
//

#include "count.cuh"

#include <thrust/sort.h>
#include <thrust/device_vector.h>

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts) {

    thrust::device_vector<int> d_Sorted = d_in;

    thrust::sort(d_Sorted.begin(), d_Sorted.end());

    size_t run_size = thrust::reduce_by_key(d_Sorted.begin(), d_Sorted.end(), thrust::constant_iterator<int>(1),
        values.begin(), counts.begin()).first - values.begin();

    values.resize(run_size);
    counts.resize(run_size);
}
