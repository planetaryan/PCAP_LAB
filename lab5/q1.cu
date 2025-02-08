#include <stdio.h>
#include <cuda_runtime.h>

// Kernel 1: Add vectors with block size as N
__global__ void vectorAddBlockSizeAsN(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Kernel 1 - Block ID: %d, Thread ID: %d\n", blockIdx.x, threadIdx.x);
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel 2: Add vectors using N threads
__global__ void vectorAddNThreads(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Kernel 2 - Block ID: %d, Thread ID: %d\n", blockIdx.x, threadIdx.x);
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel 3: Add vectors with 256 threads per block, varying number of blocks (using dim3)
__global__ void vectorAdd256ThreadsPerBlock(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Kernel 3 - Block ID: %d, Thread ID: %d\n", blockIdx.x, threadIdx.x);
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 3; // Length of the vectors (can be changed)
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

    // Kernel 1: Block size as N
    printf("Launching Kernel 1 (Block Size = N)\n");
    vectorAddBlockSizeAsN<<<1, N>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result vector C from Kernel 1:\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %d\n", i, h_C[i]);
    }

    // Kernel 2: Using N threads
    printf("Launching Kernel 2 (Using N Threads)\n");
    vectorAddNThreads<<<1, N>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result vector C from Kernel 2:\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %d\n", i, h_C[i]);
    }

    // Kernel 3: 256 threads per block, varying number of blocks (using dim3)
    printf("Launching Kernel 3 (256 Threads per Block)\n");
    int blockSize = 256;
    int numBlocks = ceil(N/256.0);  // Calculate number of blocks

    // Using dim3 for block and grid size (1D grid with 256 threads per block)
    dim3 block(blockSize);
    dim3 grid(numBlocks);

    vectorAdd256ThreadsPerBlock<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result vector C from Kernel 3:\n");
    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %d\n", i, h_C[i]);
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
// Vector B:
// B[0] = 0
// B[1] = 2
// B[2] = 4
// Launching Kernel 1 (Block Size = N)
// Kernel 1 - Block ID: 0, Thread ID: 0
// Kernel 1 - Block ID: 0, Thread ID: 1
// Kernel 1 - Block ID: 0, Thread ID: 2
// Result vector C from Kernel 1:
// C[0] = 0
// C[1] = 3
// C[2] = 6
// Launching Kernel 2 (Using N Threads)
// Kernel 2 - Block ID: 0, Thread ID: 0
// Kernel 2 - Block ID: 0, Thread ID: 1
// Kernel 2 - Block ID: 0, Thread ID: 2
// Result vector C from Kernel 2:
// C[0] = 0
// C[1] = 3
// C[2] = 6
// Launching Kernel 3 (256 Threads per Block)
// Kernel 3 - Block ID: 0, Thread ID: 224
// Kernel 3 - Block ID: 0, Thread ID: 225
// Kernel 3 - Block ID: 0, Thread ID: 226
// Kernel 3 - Block ID: 0, Thread ID: 227
// Kernel 3 - Block ID: 0, Thread ID: 228
// Kernel 3 - Block ID: 0, Thread ID: 229
// Kernel 3 - Block ID: 0, Thread ID: 230
// Kernel 3 - Block ID: 0, Thread ID: 231
// Kernel 3 - Block ID: 0, Thread ID: 232
// Kernel 3 - Block ID: 0, Thread ID: 233
// Kernel 3 - Block ID: 0, Thread ID: 234
// Kernel 3 - Block ID: 0, Thread ID: 235
// Kernel 3 - Block ID: 0, Thread ID: 236
// Kernel 3 - Block ID: 0, Thread ID: 237
// Kernel 3 - Block ID: 0, Thread ID: 238
// Kernel 3 - Block ID: 0, Thread ID: 239
// Kernel 3 - Block ID: 0, Thread ID: 240
// Kernel 3 - Block ID: 0, Thread ID: 241
// Kernel 3 - Block ID: 0, Thread ID: 242
// Kernel 3 - Block ID: 0, Thread ID: 243
// Kernel 3 - Block ID: 0, Thread ID: 244
// Kernel 3 - Block ID: 0, Thread ID: 245
// Kernel 3 - Block ID: 0, Thread ID: 246
// Kernel 3 - Block ID: 0, Thread ID: 247
// Kernel 3 - Block ID: 0, Thread ID: 248
// Kernel 3 - Block ID: 0, Thread ID: 249
// Kernel 3 - Block ID: 0, Thread ID: 250
// Kernel 3 - Block ID: 0, Thread ID: 251
// Kernel 3 - Block ID: 0, Thread ID: 252
// Kernel 3 - Block ID: 0, Thread ID: 253
// Kernel 3 - Block ID: 0, Thread ID: 254
// Kernel 3 - Block ID: 0, Thread ID: 255
// Kernel 3 - Block ID: 0, Thread ID: 96
// Kernel 3 - Block ID: 0, Thread ID: 97
// Kernel 3 - Block ID: 0, Thread ID: 98
// Kernel 3 - Block ID: 0, Thread ID: 99
// Kernel 3 - Block ID: 0, Thread ID: 100
// Kernel 3 - Block ID: 0, Thread ID: 101
// Kernel 3 - Block ID: 0, Thread ID: 102
// Kernel 3 - Block ID: 0, Thread ID: 103
// Kernel 3 - Block ID: 0, Thread ID: 104
// Kernel 3 - Block ID: 0, Thread ID: 105
// Kernel 3 - Block ID: 0, Thread ID: 106
// Kernel 3 - Block ID: 0, Thread ID: 107
// Kernel 3 - Block ID: 0, Thread ID: 108
// Kernel 3 - Block ID: 0, Thread ID: 109
// Kernel 3 - Block ID: 0, Thread ID: 110
// Kernel 3 - Block ID: 0, Thread ID: 111
// Kernel 3 - Block ID: 0, Thread ID: 112
// Kernel 3 - Block ID: 0, Thread ID: 113
// Kernel 3 - Block ID: 0, Thread ID: 114
// Kernel 3 - Block ID: 0, Thread ID: 115
// Kernel 3 - Block ID: 0, Thread ID: 116
// Kernel 3 - Block ID: 0, Thread ID: 117
// Kernel 3 - Block ID: 0, Thread ID: 118
// Kernel 3 - Block ID: 0, Thread ID: 119
// Kernel 3 - Block ID: 0, Thread ID: 120
// Kernel 3 - Block ID: 0, Thread ID: 121
// Kernel 3 - Block ID: 0, Thread ID: 122
// Kernel 3 - Block ID: 0, Thread ID: 123
// Kernel 3 - Block ID: 0, Thread ID: 124
// Kernel 3 - Block ID: 0, Thread ID: 125
// Kernel 3 - Block ID: 0, Thread ID: 126
// Kernel 3 - Block ID: 0, Thread ID: 127
// Kernel 3 - Block ID: 0, Thread ID: 160
// Kernel 3 - Block ID: 0, Thread ID: 161
// Kernel 3 - Block ID: 0, Thread ID: 162
// Kernel 3 - Block ID: 0, Thread ID: 163
// Kernel 3 - Block ID: 0, Thread ID: 164
// Kernel 3 - Block ID: 0, Thread ID: 165
// Kernel 3 - Block ID: 0, Thread ID: 166
// Kernel 3 - Block ID: 0, Thread ID: 167
// Kernel 3 - Block ID: 0, Thread ID: 168
// Kernel 3 - Block ID: 0, Thread ID: 169
// Kernel 3 - Block ID: 0, Thread ID: 170
// Kernel 3 - Block ID: 0, Thread ID: 171
// Kernel 3 - Block ID: 0, Thread ID: 172
// Kernel 3 - Block ID: 0, Thread ID: 173
// Kernel 3 - Block ID: 0, Thread ID: 174
// Kernel 3 - Block ID: 0, Thread ID: 175
// Kernel 3 - Block ID: 0, Thread ID: 176
// Kernel 3 - Block ID: 0, Thread ID: 177
// Kernel 3 - Block ID: 0, Thread ID: 178
// Kernel 3 - Block ID: 0, Thread ID: 179
// Kernel 3 - Block ID: 0, Thread ID: 180
// Kernel 3 - Block ID: 0, Thread ID: 181
// Kernel 3 - Block ID: 0, Thread ID: 182
// Kernel 3 - Block ID: 0, Thread ID: 183
// Kernel 3 - Block ID: 0, Thread ID: 184
// Kernel 3 - Block ID: 0, Thread ID: 185
// Kernel 3 - Block ID: 0, Thread ID: 186
// Kernel 3 - Block ID: 0, Thread ID: 187
// Kernel 3 - Block ID: 0, Thread ID: 188
// Kernel 3 - Block ID: 0, Thread ID: 189
// Kernel 3 - Block ID: 0, Thread ID: 190
// Kernel 3 - Block ID: 0, Thread ID: 191
// Kernel 3 - Block ID: 0, Thread ID: 192
// Kernel 3 - Block ID: 0, Thread ID: 193
// Kernel 3 - Block ID: 0, Thread ID: 194
// Kernel 3 - Block ID: 0, Thread ID: 195
// Kernel 3 - Block ID: 0, Thread ID: 196
// Kernel 3 - Block ID: 0, Thread ID: 197
// Kernel 3 - Block ID: 0, Thread ID: 198
// Kernel 3 - Block ID: 0, Thread ID: 199
// Kernel 3 - Block ID: 0, Thread ID: 200
// Kernel 3 - Block ID: 0, Thread ID: 201
// Kernel 3 - Block ID: 0, Thread ID: 202
// Kernel 3 - Block ID: 0, Thread ID: 203
// Kernel 3 - Block ID: 0, Thread ID: 204
// Kernel 3 - Block ID: 0, Thread ID: 205
// Kernel 3 - Block ID: 0, Thread ID: 206
// Kernel 3 - Block ID: 0, Thread ID: 207
// Kernel 3 - Block ID: 0, Thread ID: 208
// Kernel 3 - Block ID: 0, Thread ID: 209
// Kernel 3 - Block ID: 0, Thread ID: 210
// Kernel 3 - Block ID: 0, Thread ID: 211
// Kernel 3 - Block ID: 0, Thread ID: 212
// Kernel 3 - Block ID: 0, Thread ID: 213
// Kernel 3 - Block ID: 0, Thread ID: 214
// Kernel 3 - Block ID: 0, Thread ID: 215
// Kernel 3 - Block ID: 0, Thread ID: 216
// Kernel 3 - Block ID: 0, Thread ID: 217
// Kernel 3 - Block ID: 0, Thread ID: 218
// Kernel 3 - Block ID: 0, Thread ID: 219
// Kernel 3 - Block ID: 0, Thread ID: 220
// Kernel 3 - Block ID: 0, Thread ID: 221
// Kernel 3 - Block ID: 0, Thread ID: 222
// Kernel 3 - Block ID: 0, Thread ID: 223
// Kernel 3 - Block ID: 0, Thread ID: 0
// Kernel 3 - Block ID: 0, Thread ID: 1
// Kernel 3 - Block ID: 0, Thread ID: 2
// Kernel 3 - Block ID: 0, Thread ID: 3
// Kernel 3 - Block ID: 0, Thread ID: 4
// Kernel 3 - Block ID: 0, Thread ID: 5
// Kernel 3 - Block ID: 0, Thread ID: 6
// Kernel 3 - Block ID: 0, Thread ID: 7
// Kernel 3 - Block ID: 0, Thread ID: 8
// Kernel 3 - Block ID: 0, Thread ID: 9
// Kernel 3 - Block ID: 0, Thread ID: 10
// Kernel 3 - Block ID: 0, Thread ID: 11
// Kernel 3 - Block ID: 0, Thread ID: 12
// Kernel 3 - Block ID: 0, Thread ID: 13
// Kernel 3 - Block ID: 0, Thread ID: 14
// Kernel 3 - Block ID: 0, Thread ID: 15
// Kernel 3 - Block ID: 0, Thread ID: 16
// Kernel 3 - Block ID: 0, Thread ID: 17
// Kernel 3 - Block ID: 0, Thread ID: 18
// Kernel 3 - Block ID: 0, Thread ID: 19
// Kernel 3 - Block ID: 0, Thread ID: 20
// Kernel 3 - Block ID: 0, Thread ID: 21
// Kernel 3 - Block ID: 0, Thread ID: 22
// Kernel 3 - Block ID: 0, Thread ID: 23
// Kernel 3 - Block ID: 0, Thread ID: 24
// Kernel 3 - Block ID: 0, Thread ID: 25
// Kernel 3 - Block ID: 0, Thread ID: 26
// Kernel 3 - Block ID: 0, Thread ID: 27
// Kernel 3 - Block ID: 0, Thread ID: 28
// Kernel 3 - Block ID: 0, Thread ID: 29
// Kernel 3 - Block ID: 0, Thread ID: 30
// Kernel 3 - Block ID: 0, Thread ID: 31
// Kernel 3 - Block ID: 0, Thread ID: 128
// Kernel 3 - Block ID: 0, Thread ID: 129
// Kernel 3 - Block ID: 0, Thread ID: 130
// Kernel 3 - Block ID: 0, Thread ID: 131
// Kernel 3 - Block ID: 0, Thread ID: 132
// Kernel 3 - Block ID: 0, Thread ID: 133
// Kernel 3 - Block ID: 0, Thread ID: 134
// Kernel 3 - Block ID: 0, Thread ID: 135
// Kernel 3 - Block ID: 0, Thread ID: 136
// Kernel 3 - Block ID: 0, Thread ID: 137
// Kernel 3 - Block ID: 0, Thread ID: 138
// Kernel 3 - Block ID: 0, Thread ID: 139
// Kernel 3 - Block ID: 0, Thread ID: 140
// Kernel 3 - Block ID: 0, Thread ID: 141
// Kernel 3 - Block ID: 0, Thread ID: 142
// Kernel 3 - Block ID: 0, Thread ID: 143
// Kernel 3 - Block ID: 0, Thread ID: 144
// Kernel 3 - Block ID: 0, Thread ID: 145
// Kernel 3 - Block ID: 0, Thread ID: 146
// Kernel 3 - Block ID: 0, Thread ID: 147
// Kernel 3 - Block ID: 0, Thread ID: 148
// Kernel 3 - Block ID: 0, Thread ID: 149
// Kernel 3 - Block ID: 0, Thread ID: 150
// Kernel 3 - Block ID: 0, Thread ID: 151
// Kernel 3 - Block ID: 0, Thread ID: 152
// Kernel 3 - Block ID: 0, Thread ID: 153
// Kernel 3 - Block ID: 0, Thread ID: 154
// Kernel 3 - Block ID: 0, Thread ID: 155
// Kernel 3 - Block ID: 0, Thread ID: 156
// Kernel 3 - Block ID: 0, Thread ID: 157
// Kernel 3 - Block ID: 0, Thread ID: 158
// Kernel 3 - Block ID: 0, Thread ID: 159
// Kernel 3 - Block ID: 0, Thread ID: 64
// Kernel 3 - Block ID: 0, Thread ID: 65
// Kernel 3 - Block ID: 0, Thread ID: 66
// Kernel 3 - Block ID: 0, Thread ID: 67
// Kernel 3 - Block ID: 0, Thread ID: 68
// Kernel 3 - Block ID: 0, Thread ID: 69
// Kernel 3 - Block ID: 0, Thread ID: 70
// Kernel 3 - Block ID: 0, Thread ID: 71
// Kernel 3 - Block ID: 0, Thread ID: 72
// Kernel 3 - Block ID: 0, Thread ID: 73
// Kernel 3 - Block ID: 0, Thread ID: 74
// Kernel 3 - Block ID: 0, Thread ID: 75
// Kernel 3 - Block ID: 0, Thread ID: 76
// Kernel 3 - Block ID: 0, Thread ID: 77
// Kernel 3 - Block ID: 0, Thread ID: 78
// Kernel 3 - Block ID: 0, Thread ID: 79
// Kernel 3 - Block ID: 0, Thread ID: 80
// Kernel 3 - Block ID: 0, Thread ID: 81
// Kernel 3 - Block ID: 0, Thread ID: 82
// Kernel 3 - Block ID: 0, Thread ID: 83
// Kernel 3 - Block ID: 0, Thread ID: 84
// Kernel 3 - Block ID: 0, Thread ID: 85
// Kernel 3 - Block ID: 0, Thread ID: 86
// Kernel 3 - Block ID: 0, Thread ID: 87
// Kernel 3 - Block ID: 0, Thread ID: 88
// Kernel 3 - Block ID: 0, Thread ID: 89
// Kernel 3 - Block ID: 0, Thread ID: 90
// Kernel 3 - Block ID: 0, Thread ID: 91
// Kernel 3 - Block ID: 0, Thread ID: 92
// Kernel 3 - Block ID: 0, Thread ID: 93
// Kernel 3 - Block ID: 0, Thread ID: 94
// Kernel 3 - Block ID: 0, Thread ID: 95
// Kernel 3 - Block ID: 0, Thread ID: 32
// Kernel 3 - Block ID: 0, Thread ID: 33
// Kernel 3 - Block ID: 0, Thread ID: 34
// Kernel 3 - Block ID: 0, Thread ID: 35
// Kernel 3 - Block ID: 0, Thread ID: 36
// Kernel 3 - Block ID: 0, Thread ID: 37
// Kernel 3 - Block ID: 0, Thread ID: 38
// Kernel 3 - Block ID: 0, Thread ID: 39
// Kernel 3 - Block ID: 0, Thread ID: 40
// Kernel 3 - Block ID: 0, Thread ID: 41
// Kernel 3 - Block ID: 0, Thread ID: 42
// Kernel 3 - Block ID: 0, Thread ID: 43
// Kernel 3 - Block ID: 0, Thread ID: 44
// Kernel 3 - Block ID: 0, Thread ID: 45
// Kernel 3 - Block ID: 0, Thread ID: 46
// Kernel 3 - Block ID: 0, Thread ID: 47
// Kernel 3 - Block ID: 0, Thread ID: 48
// Kernel 3 - Block ID: 0, Thread ID: 49
// Kernel 3 - Block ID: 0, Thread ID: 50
// Kernel 3 - Block ID: 0, Thread ID: 51
// Kernel 3 - Block ID: 0, Thread ID: 52
// Kernel 3 - Block ID: 0, Thread ID: 53
// Kernel 3 - Block ID: 0, Thread ID: 54
// Kernel 3 - Block ID: 0, Thread ID: 55
// Kernel 3 - Block ID: 0, Thread ID: 56
// Kernel 3 - Block ID: 0, Thread ID: 57
// Kernel 3 - Block ID: 0, Thread ID: 58
// Kernel 3 - Block ID: 0, Thread ID: 59
// Kernel 3 - Block ID: 0, Thread ID: 60
// Kernel 3 - Block ID: 0, Thread ID: 61
// Kernel 3 - Block ID: 0, Thread ID: 62
// Kernel 3 - Block ID: 0, Thread ID: 63
// Result vector C from Kernel 3:
// C[0] = 0
// C[1] = 3
// C[2] = 6


// Grid:   [Block 0]
//             |
// Block 0: [Thread 0, Thread 1, Thread 2, Thread 3, Thread 4]
