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
    int N = 1000; // Length of the vectors (can be changed)
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

    // Define block size and calculate grid size (number of blocks)
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Calculate number of blocks to cover all elements

    // Launch kernel
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
