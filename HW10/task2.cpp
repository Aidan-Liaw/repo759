//
// Created by aidan on 10/04/2026.
//

#include <chrono>
#include <random>
#include <string>
#include <omp.h>

#include "mpi.h"

#include "optimize.h"
#include "reduce.h"

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPT's
// ChatGPT was provided the code and was asked "Is the code correct?".
// It picked up small errors like using wrong array sizes, and wrong delete statements

int main(int argc, char *argv[]) {
    std::size_t array_length = std::stoull(argv[1]);
    std::size_t thread_count = std::stoull(argv[2]);

    omp_set_num_threads(thread_count);

    float *arr = new float[2 * array_length];

    std::random_device rng_device;
    std::mt19937 generator(rng_device());

    // std::numeric_limits<float>::epsilon() is the FP epsilon
    // Which is added to ensure that 1 is included in the distribution
    // Since uniform_real_distribution is considered inclusive-exclusive
    std::uniform_real_distribution<float> distr(0,std::numeric_limits<float>::epsilon() + 10);
    for (std::size_t idx = 0; idx < 2 * array_length; ++idx) {
        arr[idx] = distr(generator);
    }

    int rank; // Process' rank
    int p; // Number of processes
    int source; // Rank of sender
    int destination; // Rank of receiver
    int tag = 0; // Tag for messages
    float res; // Storage for message
    MPI_Status status; // Return status for receive

    int rc = MPI_Init(&argc, &argv);
    fprintf(stderr, "rank startup rc=%d\n", rc);
    if (rc != MPI_SUCCESS) return rc;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;

    MPI_Bcast(arr, 2 * array_length, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // get the processor name
    if (rank == 0) { /* the macho guy */
        start = std::chrono::high_resolution_clock::now();
        res = reduce(arr, 0, array_length);
    } else if (rank == 1) { /* worker */
        start = std::chrono::high_resolution_clock::now();
        res = reduce(arr, array_length, 2 * array_length);
    }

    float global_res;
    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();

    duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(end - start);
    double temp = duration_msec.count();
    double longest_time;
    MPI_Reduce(&temp, &longest_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%f\n", global_res);
        printf("%f\n", longest_time);
    }

    MPI_Finalize();

    delete[] arr;
}
