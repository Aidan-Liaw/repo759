//
// Created by aidan on 12/03/2026.
//

#include "count.cuh"

#include <thrust/sort.h>
#include <thrust/device_vector.h>

// ACKNOWLEDGEMENT: 70% of the work is my own, and 20% of the work is ChatGPT's,
// and 10% of the work is Nvidia's article on Thrust as they used an RLE (Run-Length Encoding) example,
// which is basically this task.
// For this task, I asked ChatGPT "Is my code correct?" and provided it with my code.
// It picked up that I was trying to modify a constant vector
// (even though to my eyes it looks like a constant pointer...),
// and warned that I needed to resize the values and counts vectors like Nvidia's article does.
// Otherwise, it was satisfied with my code.

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
