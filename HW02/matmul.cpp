//
// Created by aidan on 3/02/2026.
//

#include "matmul.h"

// ACKNOWLEDGEMENT: The problem was solved by me 90% and ChatGPT 10%
// The AI was asked to check for bugs, and found that based off the
// specification, clearing the values of C should be done here instead of
// in task3.cpp

void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
	    for (unsigned int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
    std::fill_n(C, n * n, 0.0);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int k = 0; k < n; k++) {
            for (unsigned int j = 0; j < n; j++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
    std::fill_n(C, n * n, 0.0);
    for (unsigned int j = 0; j < n; j++) {
        for (unsigned int k = 0; k < n; k++) {
            for (unsigned int i = 0; i < n; i++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
	    C[i * n + j] = 0.0;
            for (unsigned int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
