#include <cuda.h>

// ACKNOWLEDGEMENT: 10% of the work is my own, 30% of the work is ChatGPT's,
// and 60% of the work is the TAs' and Instructor's.
// This is because most of the code is from the lecture slides.
// ChatGPT was asked "Is the code correct?" mutliple times.

// ChatGPT's main contribution was fixing dumb typos
// (like blindly copying the assignment for column to row without changing the member to y)
// or more serious mistakes like leaving the function early which would cause a deadlock as
// early-returned functions would never call __syncthreads()
// and I forgot that all threads must reach __syncthreads() at some point in their execution for deadlock to not occur
// I later made the mistake of allowing for threads whose indexes would go over the matrix size to exit early,
// which ChatGPT caught, which again would cause some threads to exit early.
// Reading the speaker notes of Lecture 11 also helped me realise this later on.
// It also caught me trying to 2D index a pointer.
// It also told me that all threads must write to shared memory to 0 pad areas that are invalid, but still accessible
// to other threads

// I should however acknowledge, that this code was based off HW04's matmul.cu and stencil.cu,
// which ChatGPT contributed around 70% of the work. You may choose to follow that number instead
// (and the ramifications of such heavy usage as well).

__global__ void matmul_1_kernel(const int *A, const int *B, int *C, unsigned int n) {
        // Block index
    int bx = blockIdx.x; //the B (and C) matrix sub-block column index
    int by = blockIdx.y; //the A (and C) matrix sub-block row index

    // Thread index
    int tx = threadIdx.x; //the column index in the sub-block
    int ty = threadIdx.y; //the row index in the sub-block

    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int tile = blockDim.x; // Can also use blockDim.y, it doesn't matter

    extern __shared__ int shared_memory[];
    int *As = shared_memory;
    int *Bs = shared_memory + tile * tile; // Basically shared_memory + sizeof(As)

    int Csub = 0;
    // Shared memory for the sub-matrices (tiles) of and B

    // Loop over all the sub-matrices (tiles) of A and B required to
    // compute the block sub-matrix; moving in A left to right in
    // a row, and in B from top to bottom in a column

    // Upper bound is the smallest number of tiles to cover, ceil(n/tileCount)
    // Ensures that all threads iterate through the same number of cycles
    for (int tileIdx = 0; tileIdx < (n + blockDim.x - 1) / blockDim.x; tileIdx++) {
        int Acolumn = tileIdx * tile + tx; // k index for A
        int Brow = tileIdx * tile + ty; // k index for B

        // Load tiles from global memory into shared memory; each
        // thread loads one element of the two tiles from A & B
        As[ty * tile + tx] = row < n && Acolumn < n ? A[row * n  + Acolumn] : 0;
        Bs[ty * tile + tx] = column < n && Brow < n ? B[Brow * n + column]  : 0;

        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        // Each thread in this block computes one element
        // of the block sub-matrix (tile). Thread with indexes
        // ty and tx computes in this tile the entry [ty][tx].
        for (int k = 0; k < tile; ++k) {
            Csub += As[ty * tile + k] * Bs[k * tile + tx];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory;
    // each thread writes one element
    if (row < n && column < n) {
         C[row * n + column] = Csub;
    }
}

__global__ void matmul_2_kernel(const float *A, const float *B, float *C, unsigned int n) {
     // Block index
    int bx = blockIdx.x; //the B (and C) matrix sub-block column index
    int by = blockIdx.y; //the A (and C) matrix sub-block row index

    // Thread index
    int tx = threadIdx.x; //the column index in the sub-block
    int ty = threadIdx.y; //the row index in the sub-block

    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int tile = blockDim.x; // Can also use blockDim.y, it doesn't matter

    extern __shared__ float shared_memory[];
    float *As = shared_memory;
    float *Bs = shared_memory + tile * tile; // Basically shared_memory + sizeof(sA)

    float Csub = 0;
    // Shared memory for the sub-matrices (tiles) of and B

    // Loop over all the sub-matrices (tiles) of A and B required to
    // compute the block sub-matrix; moving in A left to right in
    // a row, and in B from top to bottom in a column

    // Upper bound is the smallest number of tiles to cover, ceil(n/tileCount)
    // Ensures that all threads iterate through the same number of cycles
    for (int tileIdx = 0; tileIdx < (n + blockDim.x - 1) / blockDim.x; tileIdx++) {
        int Acolumn = tileIdx * tile + tx; // k index for A
        int Brow = tileIdx * tile + ty; // k index for B

        // Load tiles from global memory into shared memory; each
        // thread loads one element of the two tiles from A & B
        As[ty * tile + tx] = row < n && Acolumn < n ? A[row * n  + Acolumn] : 0;
        Bs[ty * tile + tx] = column < n && Brow < n ? B[Brow * n + column]  : 0;

        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        // Each thread in this block computes one element
        // of the block sub-matrix (tile). Thread with indexes
        // ty and tx computes in this tile the entry [ty][tx].
        for (int k = 0; k < tile; ++k) {
            Csub += As[ty * tile + k] * Bs[k * tile + tx];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory;
    // each thread writes one element
    if (row < n && column < n) {
         C[row * n + column] = Csub;
    }
}

__global__ void matmul_3_kernel(const double *A, const double *B, double *C, unsigned int n) {
    // Block index
    int bx = blockIdx.x; //the B (and C) matrix sub-block column index
    int by = blockIdx.y; //the A (and C) matrix sub-block row index

    // Thread index
    int tx = threadIdx.x; //the column index in the sub-block
    int ty = threadIdx.y; //the row index in the sub-block

    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int tile = blockDim.x; // Can also use blockDim.y, it doesn't matter

    extern __shared__ double shared_memory[];
    double *As = shared_memory;
    double *Bs = shared_memory + tile * tile; // Basically shared_memory + sizeof(sA)

    double Csub = 0;
    // Shared memory for the sub-matrices (tiles) of and B

    // Loop over all the sub-matrices (tiles) of A and B required to
    // compute the block sub-matrix; moving in A left to right in
    // a row, and in B from top to bottom in a column

    // Upper bound is the smallest number of tiles to cover, ceil(n/tileCount)
    // Ensures that all threads iterate through the same number of cycles
    for (int tileIdx = 0; tileIdx < (n + blockDim.x - 1) / blockDim.x; tileIdx++) {
        int Acolumn = tileIdx * tile + tx; // k index for A
        int Brow = tileIdx * tile + ty; // k index for B

        // Load tiles from global memory into shared memory; each
        // thread loads one element of the two tiles from A & B
        As[ty * tile + tx] = row < n && Acolumn < n ? A[row * n  + Acolumn] : 0;
        Bs[ty * tile + tx] = column < n && Brow < n ? B[Brow * n + column]  : 0;

        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        // Each thread in this block computes one element
        // of the block sub-matrix (tile). Thread with indexes
        // ty and tx computes in this tile the entry [ty][tx].
        for (int k = 0; k < tile; ++k) {
            Csub += As[ty * tile + k] * Bs[k * tile + tx];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory;
    // each thread writes one element
    if (row < n && column < n) {
         C[row * n + column] = Csub;
    }
}


__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    size_t shared_memory_size = (2 * block_dim *block_dim) * sizeof(int);

    matmul_1_kernel<<<dimGrid, dimBlock, shared_memory_size>>>(A, B, C, n);

    cudaDeviceSynchronize();
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    size_t shared_memory_size = (2 * block_dim *block_dim) * sizeof(float);

    matmul_2_kernel<<<dimGrid, dimBlock, shared_memory_size>>>(A, B, C, n);

    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    size_t shared_memory_size = (2 * block_dim *block_dim) * sizeof(double);

    matmul_3_kernel<<<dimGrid, dimBlock, shared_memory_size>>>(A, B, C, n);

    cudaDeviceSynchronize();
}