//
// Created by aidan on 26/03/2026.
//

#include <cstdio>
#include <chrono>
#include <iostream>
#include <string>
#include <limits>

#include "mpi.h"

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPT's
// ChatGPT was provided the code and was asked "Is the code correct?".
// It picked up that I had used MPI_FLOAT instead of MPI_DOUBLE


int main(int argc, char *argv[]) {
    std::size_t arrayLength = std::stoull(argv[1]);

    int rank; // Process' rank
    int p; // Number of processes
    int source; // Rank of sender
    int destination; // Rank of receiver
    int tag = 0; // Tag for messages
    float *message_1 = new float[arrayLength]; // Storage for message
    float *message_2 = new float[arrayLength]; // Storage for message
    MPI_Status status; // Return status for receive

    for (std::size_t idx = 0; idx < arrayLength; ++idx) {
        message_1[idx] = (long) idx;
        message_2[idx] = (long) arrayLength - 1 - idx;
    }

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_msec;
    double worker_duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // get the processor name
    if (rank == 0) { /* the macho guy */
        destination = 1;
        source = 1;

        start = std::chrono::high_resolution_clock::now();
        MPI_Send(message_1, arrayLength, MPI_FLOAT, destination, tag, MPI_COMM_WORLD);
        MPI_Recv(message_1, arrayLength, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
        end = std::chrono::high_resolution_clock::now();
        duration_msec = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        MPI_Recv(&worker_duration, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);

        std::printf("%f\n", (duration_msec.count() + worker_duration));
    } else if (rank == 1) { /* worker */
        destination = 0;
        source = 0;

        start = std::chrono::high_resolution_clock::now();
        MPI_Recv(message_2, arrayLength, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
        MPI_Send(message_2, arrayLength, MPI_FLOAT, destination, tag, MPI_COMM_WORLD);
        end = std::chrono::high_resolution_clock::now();

        worker_duration =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

        MPI_Send(&worker_duration, 1, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    delete [] message_1;
    delete [] message_2;
}
