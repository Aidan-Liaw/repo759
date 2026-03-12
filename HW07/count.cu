//
// Created by aidan on 12/03/2026.
//

#include "count.cuh"

#include <thrust/sort.h>
#include <thrust/device_vector.h>

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts) {

    thrust::sort(d_in.begin(), d_in.end());

    thrust::reduce_by_key(d_in.begin(), d_in.end(), thrust::constant_iterator<int>(1),
        values.begin(), counts.begin());
}
