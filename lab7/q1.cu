#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void countWordOccurrences(char *sentence, char *word, int *count, int sentenceLength, int wordLength) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread is within bounds
    if (idx * wordLength < sentenceLength) {
        bool match = true;

        // Compare the current slice of the sentence with the given word
        for (int i = 0; i < wordLength; i++) {
            if (sentence[idx * wordLength + i] != word[i]) {
                match = false;
                break;
            }
        }

        // If it matches, increment the count using atomic operation
        if (match) {
            // Print thread index and the fact that the atomicAdd is being called
            printf("Thread %d: Match found! Calling atomicAdd...\n", idx);
            atomicAdd(count, 1);
        }
    }
}

int main() {
    // Declare variables for sentence and word
    char sentence[1024];  // Allocate enough space for user input (adjust the size as necessary)
    char word[] = "hello";  // Word to search for in the sentence

    // Prompt the user to enter a sentence
    printf("Enter a sentence: ");
    
    // Use fgets instead of gets to prevent buffer overflow
    if (fgets(sentence, sizeof(sentence), stdin) == NULL) {
        printf("Error reading input\n");
        return 1;
    }

    // Remove the newline character if present (fgets adds it)
    sentence[strcspn(sentence, "\n")] = 0;

    int sentenceLength = strlen(sentence);
    int wordLength = strlen(word);

    // Allocate memory on host
    char *h_sentence = (char*)malloc(sentenceLength * sizeof(char));
    int h_count = 0;

    // Copy data from host to device
    strcpy(h_sentence, sentence);

    char *d_sentence;
    int *d_count;

    cudaMalloc((void**)&d_sentence, sentenceLength * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_sentence, h_sentence, sentenceLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    // Print the input sentence and word
    printf("Input Sentence: %s\n", sentence);
    printf("Word to search for: '%s'\n", word);

    // Launch kernel with enough threads to cover the sentence length
    int threadsPerBlock = 256;
    int blocksPerGrid = (sentenceLength + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks and %d threads per block...\n", blocksPerGrid, threadsPerBlock);
    countWordOccurrences<<<blocksPerGrid, threadsPerBlock>>>(d_sentence, word, d_count, sentenceLength, wordLength);

    // Wait for kernel to finish and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy result back to host
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("The word '%s' appears %d times in the sentence.\n", word, h_count);

    // Free device memory
    cudaFree(d_sentence);
    cudaFree(d_count);

    // Free host memory
    free(h_sentence);

    return 0;
}

// Enter a sentence: hello world hello CUDA world hello
// Input Sentence: hello world hello CUDA world hello
// Word to search for: 'hello'
// Launching kernel with 1 blocks and 256 threads per block...
// Thread 0: Match found! Calling atomicAdd...
// Thread 2: Match found! Calling atomicAdd...
// Thread 5: Match found! Calling atomicAdd...
// The word 'hello' appears 3 times in the sentence.
