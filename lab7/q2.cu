#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void generateRS(char *S, char *RS, int S_length, int RS_length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread handles one position in RS
    if (idx < RS_length) {
        int copy_idx = idx;

        // Ensure the index is within the bounds of string S
        if (copy_idx < S_length) {
            // Copy the character to RS
            RS[idx] = S[copy_idx];

            // Print details to show which character each thread is copying
            printf("Thread %d: Copying character '%c' from S[%d] to RS[%d]\n", idx, S[copy_idx], copy_idx, idx);
        }
    }
}

int main() {
    // Input string S
    const char *S = "PCAP";
    int S_length = strlen(S);
    
    // Length of the resultant string RS
    int RS_length = S_length * 2 - 1;

    // Allocate memory on host
    char *h_S = (char*)malloc(S_length * sizeof(char));
    char *h_RS = (char*)malloc(RS_length * sizeof(char));

    // Copy the input string S to host memory
    strcpy(h_S, S);

    // Allocate memory on device
    char *d_S, *d_RS;
    cudaMalloc((void**)&d_S, S_length * sizeof(char));
    cudaMalloc((void**)&d_RS, RS_length * sizeof(char));

    // Copy input string to device
    cudaMemcpy(d_S, h_S, S_length * sizeof(char), cudaMemcpyHostToDevice);

    // Print the input string S
    printf("Input string S: %s\n", S);
    printf("Generating output string RS...\n");

    // Set up CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (RS_length + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to generate RS
    generateRS<<<blocksPerGrid, threadsPerBlock>>>(d_S, d_RS, S_length, RS_length);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_RS, d_RS, RS_length * sizeof(char), cudaMemcpyDeviceToHost);

    // Null terminate the string RS for proper printing
    h_RS[RS_length] = '\0';

    // Print the generated output string RS
    printf("Output string RS: %s\n", h_RS);

    // Free device memory
    cudaFree(d_S);
    cudaFree(d_RS);

    // Free host memory
    free(h_S);
    free(h_RS);

    return 0;
}

// Input string S: PCAP
// Generating output string RS...
// Thread 0: Copying character 'P' from S[0] to RS[0]
// Thread 1: Copying character 'P' from S[0] to RS[1]
// Thread 2: Copying character 'P' from S[0] to RS[2]
// Thread 3: Copying character 'A' from S[1] to RS[3]
// Thread 4: Copying character 'A' from S[1] to RS[4]
// Thread 5: Copying character 'A' from S[1] to RS[5]
// Thread 6: Copying character 'C' from S[2] to RS[6]
// Thread 7: Copying character 'C' from S[2] to RS[7]
// Thread 8: Copying character 'C' from S[2] to RS[8]
// Thread 9: Copying character 'P' from S[3] to RS[9]
// Thread 10: Copying character 'P' from S[3] to RS[10]
// Thread 11: Copying character 'P' from S[3] to RS[11]
// Output string RS: PCAPPCAPPC
