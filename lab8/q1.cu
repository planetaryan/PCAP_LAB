#include <stdio.h>
#include <cuda_runtime.h>

#define N 4 // Size of the matrix (4x4)

// Matrix addition kernel: Each row of the resultant matrix is computed by one thread
__global__ void matrixAddRowByRow(int *A, int *B, int *C, int width) {
    int row = threadIdx.x;
    if (row < width) {
        for (int col = 0; col < width; col++) {
            C[row * width + col] = A[row * width + col] + B[row * width + col];
        }
    }
}

// Matrix addition kernel: Each column of the resultant matrix is computed by one thread
__global__ void matrixAddColumnByColumn(int *A, int *B, int *C, int width) {
    int col = threadIdx.x;
    if (col < width) {
        for (int row = 0; row < width; row++) {
            C[row * width + col] = A[row * width + col] + B[row * width + col];
        }
    }
}

// Matrix addition kernel: Each element of the resultant matrix is computed by one thread
__global__ void matrixAddElementByElement(int *A, int *B, int *C, int width) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    if (row < width && col < width) {
        C[row * width + col] = A[row * width + col] + B[row * width + col];
    }
}

void printMatrix(int *mat, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", mat[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int h_A[N * N], h_B[N * N], h_C[N * N];
    int *d_A, *d_B, *d_C;

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Case (a): Each row of the resultant matrix is computed by one thread
    matrixAddRowByRow<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix addition (row by row):\n");
    printMatrix(h_C, N);

    // Case (b): Each column of the resultant matrix is computed by one thread
    matrixAddColumnByColumn<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nMatrix addition (column by column):\n");
    printMatrix(h_C, N);

    // Case (c): Each element of the resultant matrix is computed by one thread
    dim3 threadsPerBlock(N, N);
    matrixAddElementByElement<<<1, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nMatrix addition (element by element):\n");
    printMatrix(h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
