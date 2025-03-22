#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to perform Sparse Matrix-Vector multiplication
__global__ void spmv_kernel(int *d_values, int *d_columns, int *d_row_ptr, int *d_x, int *d_y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        int start = d_row_ptr[row];
        int end = d_row_ptr[row + 1];
        
        int sum = 0;
        for (int j = start; j < end; ++j) {
            sum += d_values[j] * d_x[d_columns[j]];
        }
        d_y[row] = sum;
    }
}

int main() {
    // Example Sparse Matrix (4x4)
    // CSR Format:
    // Values (d_values): [10, 20, 30, 40, 50, 60]
    // Columns (d_columns): [0, 1, 2, 0, 2, 3]
    // Row pointers (d_row_ptr): [0, 2, 4, 5, 6] (indicates the start and end of each row)
    // Vector x: [1, 2, 3, 4]
    
    int num_rows = 4;
    int num_cols = 4;
    int num_non_zero = 6;

    // Host data
    int h_values[] = {10, 20, 30, 40, 50, 60};
    int h_columns[] = {0, 1, 2, 0, 2, 3};
    int h_row_ptr[] = {0, 2, 4, 5, 6};
    int h_x[] = {1, 2, 3, 4};
    int h_y[4] = {0};  // Resultant vector

    // Print the input sparse matrix in CSR format
    printf("Sparse Matrix in CSR format:\n");
    printf("Values: ");
    for (int i = 0; i < num_non_zero; i++) {
        printf("%d ", h_values[i]);
    }
    printf("\nColumns: ");
    for (int i = 0; i < num_non_zero; i++) {
        printf("%d ", h_columns[i]);
    }
    printf("\nRow pointers: ");
    for (int i = 0; i <= num_rows; i++) {
        printf("%d ", h_row_ptr[i]);
    }
    printf("\n");

    // Print the input vector x
    printf("Input vector x: ");
    for (int i = 0; i < num_cols; i++) {
        printf("%d ", h_x[i]);
    }
    printf("\n");

    // Device data pointers
    int *d_values, *d_columns, *d_row_ptr, *d_x, *d_y;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_values, num_non_zero * sizeof(int));
    cudaMalloc((void**)&d_columns, num_non_zero * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, num_cols * sizeof(int));
    cudaMalloc((void**)&d_y, num_rows * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_values, h_values, num_non_zero * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, h_columns, num_non_zero * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with enough threads to handle the rows
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (num_rows + blockSize - 1) / blockSize;  // Number of blocks required

    printf("Launching kernel with %d blocks and %d threads per block.\n", numBlocks, blockSize);

    spmv_kernel<<<numBlocks, blockSize>>>(d_values, d_columns, d_row_ptr, d_x, d_y, num_rows);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result from device to host
    cudaMemcpy(h_y, d_y, num_rows * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result (Output vector y)
    printf("Resultant vector y: [ ");
    for (int i = 0; i < num_rows; i++) {
        printf("%d ", h_y[i]);
    }
    printf("]\n");

    // Free the allocated memory on device
    cudaFree(d_values);
    cudaFree(d_columns);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
// Sparse Matrix in CSR format:
// Values: 10 20 30 40 50 60 
// Columns: 0 1 2 0 2 3 
// Row pointers: 0 2 4 5 6 
// Input vector x: 1 2 3 4 
// Launching kernel with 1 blocks and 256 threads per block.
// Resultant vector y: [ 50 130 150 240 ]