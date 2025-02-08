#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to add two vectors
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 10; // Length of the vectors (can be changed as needed)
    size_t size = N * sizeof(int);

    // Allocate memory for vectors on host
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize vectors A and B with some values
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;         // Vector A: 0, 1, 2, ..., N-1
        h_B[i] = 2 * i;     // Vector B: 0, 2, 4, ..., 2*(N-1)
    }

    // Allocate memory for vectors on device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Print the input vectors A and B
    printf("Vector A:\n");
    for (int i = 0; i < N; ++i) {
        printf("A[%d] = %d\n", i, h_A[i]);
    }

    printf("Vector B:\n");
    for (int i = 0; i < N; ++i) {
        printf("B[%d] = %d\n", i, h_B[i]);
    }

    // Launch kernel with a grid of blocks, each containing N threads
    // Let's choose a reasonable block size, for example, 256 threads per block
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Calculate number of blocks

    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the result vector after kernel launch
    printf("Result vector C after kernel launch:\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %d\n", i, h_C[i]);
    }

    // Verify the result
    int success = 1;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Mismatch at index %d: expected %d but got %d\n", i, h_A[i] + h_B[i], h_C[i]);
            success = 0;
            break;
        }
    }

    if (success) {
        printf("Vectors added successfully!\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


// Vector A:
// A[0] = 0
// A[1] = 1
// A[2] = 2
// A[3] = 3
// A[4] = 4
// A[5] = 5
// A[6] = 6
// A[7] = 7
// A[8] = 8
// A[9] = 9
// Vector B:
// B[0] = 0
// B[1] = 2
// B[2] = 4
// B[3] = 6
// B[4] = 8
// B[5] = 10
// B[6] = 12
// B[7] = 14
// B[8] = 16
// B[9] = 18
// Result vector C after kernel launch:
// C[0] = 0
// C[1] = 3
// C[2] = 6
// C[3] = 9
// C[4] = 12
// C[5] = 15
// C[6] = 18
// C[7] = 21
// C[8] = 24
// C[9] = 27
// Vectors added successfully!