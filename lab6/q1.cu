#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolutionKernel(float *d_N, float *d_M, float *d_P, int width, int mask_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Thread index
    int half_mask = mask_width / 2;  // To handle the convolution for center-aligned mask

    if (idx >= half_mask && idx < (width - half_mask)) {
        float sum = 0.0f;

        // Print the convolution for this index
        printf("\nConvolution at idx %d:\n", idx);
        printf("Input Elements (N) and Mask Elements (M) being multiplied:\n");

        // Perform the convolution
        for (int j = 0; j < mask_width; j++) {
            int inputIdx = idx + j - half_mask;
            printf("N[%d] = %f * M[%d] = %f\n", inputIdx, d_N[inputIdx], j, d_M[j]);
            sum += d_N[inputIdx] * d_M[j];
        }

        // Store the result in the output array
        d_P[idx] = sum;

        printf("Sum (Result for P[%d]) = %f\n", idx, sum);
    }
}

int main() {
    // Size of the input and mask
    int width = 10;  // Length of input array N
    int mask_width = 3;  // Length of mask array M

    // Host arrays
    float h_N[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  // Input array N
    float h_M[] = {0.2, 0.5, 0.2};  // Mask array M
    float h_P[width];  // Output array P

    // Device arrays
    float *d_N, *d_M, *d_P;

    // Allocate memory on the device
    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_M, mask_width * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, mask_width * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;  // Number of threads per block
    int gridSize = (width + blockSize - 1) / blockSize;  // Number of blocks

    // Launch the convolution kernel
    convolutionKernel<<<gridSize, blockSize>>>(d_N, d_M, d_P, width, mask_width);

    // Wait for the kernel to finish (synchronization)
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Input Array N: ");
    for (int i = 0; i < width; i++) {
        printf("%f ", h_N[i]);
    }
    printf("\n");

    printf("Mask Array M: ");
    for (int i = 0; i < mask_width; i++) {
        printf("%f ", h_M[i]);
    }
    printf("\n");

    printf("Output Array P: ");
    for (int i = 0; i < width; i++) {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
