#include <stdio.h>
#include <cuda_runtime.h>

__global__ void oddPhaseKernel(int *d_arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Check if the thread is handling an odd-indexed pair
    if (idx % 2 == 1 && idx < n - 1) {
        // Compare adjacent elements at odd-even indices
        if (d_arr[idx] > d_arr[idx + 1]) {
            // Swap if necessary
            int temp = d_arr[idx];
            d_arr[idx] = d_arr[idx + 1];
            d_arr[idx + 1] = temp;
        }
    }
}

__global__ void evenPhaseKernel(int *d_arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Check if the thread is handling an even-indexed pair
    if (idx % 2 == 0 && idx < n - 1) {
        // Compare adjacent elements at even-odd indices
        if (d_arr[idx] > d_arr[idx + 1]) {
            // Swap if necessary
            int temp = d_arr[idx];
            d_arr[idx] = d_arr[idx + 1];
            d_arr[idx + 1] = temp;
        }
    }
}

void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void oddEvenSort(int *arr, int n) {
    int *d_arr;

    // Allocate memory on the device
    cudaMalloc((void**)&d_arr, n * sizeof(int));

    // Copy the array to the device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Perform the odd-even transposition sort
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            // Odd phase
            oddPhaseKernel<<<gridSize, blockSize>>>(d_arr, n);
        } else {
            // Even phase
            evenPhaseKernel<<<gridSize, blockSize>>>(d_arr, n);
        }

        // Wait for the kernel to finish
        cudaDeviceSynchronize();

        // Optionally print array at each phase (for debugging or visualization)
        if (phase < 10) {  // Print only first 10 phases for visibility
            int *tempArr = (int *)malloc(n * sizeof(int));
            cudaMemcpy(tempArr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
            printf("Array after phase %d: ", phase + 1);
            printArray(tempArr, n);
            free(tempArr);
        }
    }

    // Copy the sorted array back to the host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original Array: ");
    printArray(arr, n);

    // Perform odd-even transposition sort
    oddEvenSort(arr, n);

    printf("Sorted Array: ");
    printArray(arr, n);

    return 0;
}
