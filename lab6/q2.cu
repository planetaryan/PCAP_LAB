#include <stdio.h>
#include <cuda_runtime.h>

__global__ void findMinIndex(int *d_arr, int *d_minIndex, int start, int end) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread is within the valid range
    if (idx < end - start) {
        int index = start + idx;
        if (d_arr[index] < d_arr[d_minIndex[0]]) {
            d_minIndex[0] = index;
        }
    }
}

__global__ void swapElements(int *d_arr, int i, int j) {
    int temp = d_arr[i];
    d_arr[i] = d_arr[j];
    d_arr[j] = temp;
}

void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void selectionSort(int *arr, int n) {
    int *d_arr, *d_minIndex;

    // Allocate memory on the device
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_minIndex, sizeof(int));

    // Copy the array to the device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    printf("Original Array: ");
    printArray(arr, n);

    // Perform selection sort in parallel
    for (int i = 0; i < n - 1; i++) {
        // Set the minimum index for this iteration to i
        cudaMemcpy(d_minIndex, &i, sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel to find the index of the minimum element in the unsorted portion of the array
        int blockSize = 256;
        int gridSize = (n - i + blockSize - 1) / blockSize;
        findMinIndex<<<gridSize, blockSize>>>(d_arr, d_minIndex, i + 1, n);

        // Synchronize to make sure the kernel execution is finished
        cudaDeviceSynchronize();

        // Get the minimum index from device
        int minIndex;
        cudaMemcpy(&minIndex, d_minIndex, sizeof(int), cudaMemcpyDeviceToHost);

        // If the minimum element is not at the current index, swap
        if (minIndex != i) {
            // Print the elements being swapped
            printf("Swapping elements at index %d and %d -> ", i, minIndex);
            printf("arr[%d] = %d, arr[%d] = %d\n", i, arr[i], minIndex, arr[minIndex]);

            swapElements<<<1, 1>>>(d_arr, i, minIndex);
            cudaDeviceSynchronize();  // Ensure the swap is completed before proceeding
        }

        // Copy the array back to host to print the intermediate array
        cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Array after pass %d: ", (i + 1));
        printArray(arr, n);
    }

    // Copy the sorted array back to the host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_minIndex);
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);

    // Perform parallel selection sort
    selectionSort(arr, n);

    printf("Sorted Array: ");
    printArray(arr, n);

    return 0;
}
