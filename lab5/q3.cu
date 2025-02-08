#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel to compute sine of angles
__global__ void computeSine(float *input, float *output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Compute Sine - Block ID: %d, Thread ID: %d\n", blockIdx.x, threadIdx.x);

    // Check if the thread index is within bounds
    if (idx < N) {
        output[idx] = sinf(input[idx]);  // Compute sine of the angle in radians
    }
}

int main() {
    int N = 3;  // Length of the array (can be changed)
    size_t size = N * sizeof(float);

    // Allocate memory for input and output arrays on the host
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize the input array with some angle values in radians
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 0.1f;  // Angles from 0, 0.1, 0.2, ..., N*0.1
    }

    // Allocate memory for input and output arrays on the device
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy the input array from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;  // Number of threads per block
    int numBlocks = ceil(N/256.0);  // Number of blocks needed

    // Using dim3 for block and grid size (1D grid with 256 threads per block)
    dim3 block(blockSize);
    dim3 grid(numBlocks);

    // Launch the kernel to compute sine values
    computeSine<<<grid, block>>>(d_input, d_output, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the output array from device to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the input angles and their sine values
    printf("Angles (radians) and their sine values:\n");
    for (int i = 0; i < N; i++) {
        printf("Angle: %.2f, Sine: %.4f\n", h_input[i], h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}


// Compute Sine - Block ID: 0, Thread ID: 64
// Compute Sine - Block ID: 0, Thread ID: 65
// Compute Sine - Block ID: 0, Thread ID: 66
// Compute Sine - Block ID: 0, Thread ID: 67
// Compute Sine - Block ID: 0, Thread ID: 68
// Compute Sine - Block ID: 0, Thread ID: 69
// Compute Sine - Block ID: 0, Thread ID: 70
// Compute Sine - Block ID: 0, Thread ID: 71
// Compute Sine - Block ID: 0, Thread ID: 72
// Compute Sine - Block ID: 0, Thread ID: 73
// Compute Sine - Block ID: 0, Thread ID: 74
// Compute Sine - Block ID: 0, Thread ID: 75
// Compute Sine - Block ID: 0, Thread ID: 76
// Compute Sine - Block ID: 0, Thread ID: 77
// Compute Sine - Block ID: 0, Thread ID: 78
// Compute Sine - Block ID: 0, Thread ID: 79
// Compute Sine - Block ID: 0, Thread ID: 80
// Compute Sine - Block ID: 0, Thread ID: 81
// Compute Sine - Block ID: 0, Thread ID: 82
// Compute Sine - Block ID: 0, Thread ID: 83
// Compute Sine - Block ID: 0, Thread ID: 84
// Compute Sine - Block ID: 0, Thread ID: 85
// Compute Sine - Block ID: 0, Thread ID: 86
// Compute Sine - Block ID: 0, Thread ID: 87
// Compute Sine - Block ID: 0, Thread ID: 88
// Compute Sine - Block ID: 0, Thread ID: 89
// Compute Sine - Block ID: 0, Thread ID: 90
// Compute Sine - Block ID: 0, Thread ID: 91
// Compute Sine - Block ID: 0, Thread ID: 92
// Compute Sine - Block ID: 0, Thread ID: 93
// Compute Sine - Block ID: 0, Thread ID: 94
// Compute Sine - Block ID: 0, Thread ID: 95
// Compute Sine - Block ID: 0, Thread ID: 192
// Compute Sine - Block ID: 0, Thread ID: 193
// Compute Sine - Block ID: 0, Thread ID: 194
// Compute Sine - Block ID: 0, Thread ID: 195
// Compute Sine - Block ID: 0, Thread ID: 196
// Compute Sine - Block ID: 0, Thread ID: 197
// Compute Sine - Block ID: 0, Thread ID: 198
// Compute Sine - Block ID: 0, Thread ID: 199
// Compute Sine - Block ID: 0, Thread ID: 200
// Compute Sine - Block ID: 0, Thread ID: 201
// Compute Sine - Block ID: 0, Thread ID: 202
// Compute Sine - Block ID: 0, Thread ID: 203
// Compute Sine - Block ID: 0, Thread ID: 204
// Compute Sine - Block ID: 0, Thread ID: 205
// Compute Sine - Block ID: 0, Thread ID: 206
// Compute Sine - Block ID: 0, Thread ID: 207
// Compute Sine - Block ID: 0, Thread ID: 208
// Compute Sine - Block ID: 0, Thread ID: 209
// Compute Sine - Block ID: 0, Thread ID: 210
// Compute Sine - Block ID: 0, Thread ID: 211
// Compute Sine - Block ID: 0, Thread ID: 212
// Compute Sine - Block ID: 0, Thread ID: 213
// Compute Sine - Block ID: 0, Thread ID: 214
// Compute Sine - Block ID: 0, Thread ID: 215
// Compute Sine - Block ID: 0, Thread ID: 216
// Compute Sine - Block ID: 0, Thread ID: 217
// Compute Sine - Block ID: 0, Thread ID: 218
// Compute Sine - Block ID: 0, Thread ID: 219
// Compute Sine - Block ID: 0, Thread ID: 220
// Compute Sine - Block ID: 0, Thread ID: 221
// Compute Sine - Block ID: 0, Thread ID: 222
// Compute Sine - Block ID: 0, Thread ID: 223
// Compute Sine - Block ID: 0, Thread ID: 96
// Compute Sine - Block ID: 0, Thread ID: 97
// Compute Sine - Block ID: 0, Thread ID: 98
// Compute Sine - Block ID: 0, Thread ID: 99
// Compute Sine - Block ID: 0, Thread ID: 100
// Compute Sine - Block ID: 0, Thread ID: 101
// Compute Sine - Block ID: 0, Thread ID: 102
// Compute Sine - Block ID: 0, Thread ID: 103
// Compute Sine - Block ID: 0, Thread ID: 104
// Compute Sine - Block ID: 0, Thread ID: 105
// Compute Sine - Block ID: 0, Thread ID: 106
// Compute Sine - Block ID: 0, Thread ID: 107
// Compute Sine - Block ID: 0, Thread ID: 108
// Compute Sine - Block ID: 0, Thread ID: 109
// Compute Sine - Block ID: 0, Thread ID: 110
// Compute Sine - Block ID: 0, Thread ID: 111
// Compute Sine - Block ID: 0, Thread ID: 112
// Compute Sine - Block ID: 0, Thread ID: 113
// Compute Sine - Block ID: 0, Thread ID: 114
// Compute Sine - Block ID: 0, Thread ID: 115
// Compute Sine - Block ID: 0, Thread ID: 116
// Compute Sine - Block ID: 0, Thread ID: 117
// Compute Sine - Block ID: 0, Thread ID: 118
// Compute Sine - Block ID: 0, Thread ID: 119
// Compute Sine - Block ID: 0, Thread ID: 120
// Compute Sine - Block ID: 0, Thread ID: 121
// Compute Sine - Block ID: 0, Thread ID: 122
// Compute Sine - Block ID: 0, Thread ID: 123
// Compute Sine - Block ID: 0, Thread ID: 124
// Compute Sine - Block ID: 0, Thread ID: 125
// Compute Sine - Block ID: 0, Thread ID: 126
// Compute Sine - Block ID: 0, Thread ID: 127
// Compute Sine - Block ID: 0, Thread ID: 0
// Compute Sine - Block ID: 0, Thread ID: 1
// Compute Sine - Block ID: 0, Thread ID: 2
// Compute Sine - Block ID: 0, Thread ID: 3
// Compute Sine - Block ID: 0, Thread ID: 4
// Compute Sine - Block ID: 0, Thread ID: 5
// Compute Sine - Block ID: 0, Thread ID: 6
// Compute Sine - Block ID: 0, Thread ID: 7
// Compute Sine - Block ID: 0, Thread ID: 8
// Compute Sine - Block ID: 0, Thread ID: 9
// Compute Sine - Block ID: 0, Thread ID: 10
// Compute Sine - Block ID: 0, Thread ID: 11
// Compute Sine - Block ID: 0, Thread ID: 12
// Compute Sine - Block ID: 0, Thread ID: 13
// Compute Sine - Block ID: 0, Thread ID: 14
// Compute Sine - Block ID: 0, Thread ID: 15
// Compute Sine - Block ID: 0, Thread ID: 16
// Compute Sine - Block ID: 0, Thread ID: 17
// Compute Sine - Block ID: 0, Thread ID: 18
// Compute Sine - Block ID: 0, Thread ID: 19
// Compute Sine - Block ID: 0, Thread ID: 20
// Compute Sine - Block ID: 0, Thread ID: 21
// Compute Sine - Block ID: 0, Thread ID: 22
// Compute Sine - Block ID: 0, Thread ID: 23
// Compute Sine - Block ID: 0, Thread ID: 24
// Compute Sine - Block ID: 0, Thread ID: 25
// Compute Sine - Block ID: 0, Thread ID: 26
// Compute Sine - Block ID: 0, Thread ID: 27
// Compute Sine - Block ID: 0, Thread ID: 28
// Compute Sine - Block ID: 0, Thread ID: 29
// Compute Sine - Block ID: 0, Thread ID: 30
// Compute Sine - Block ID: 0, Thread ID: 31
// Compute Sine - Block ID: 0, Thread ID: 224
// Compute Sine - Block ID: 0, Thread ID: 225
// Compute Sine - Block ID: 0, Thread ID: 226
// Compute Sine - Block ID: 0, Thread ID: 227
// Compute Sine - Block ID: 0, Thread ID: 228
// Compute Sine - Block ID: 0, Thread ID: 229
// Compute Sine - Block ID: 0, Thread ID: 230
// Compute Sine - Block ID: 0, Thread ID: 231
// Compute Sine - Block ID: 0, Thread ID: 232
// Compute Sine - Block ID: 0, Thread ID: 233
// Compute Sine - Block ID: 0, Thread ID: 234
// Compute Sine - Block ID: 0, Thread ID: 235
// Compute Sine - Block ID: 0, Thread ID: 236
// Compute Sine - Block ID: 0, Thread ID: 237
// Compute Sine - Block ID: 0, Thread ID: 238
// Compute Sine - Block ID: 0, Thread ID: 239
// Compute Sine - Block ID: 0, Thread ID: 240
// Compute Sine - Block ID: 0, Thread ID: 241
// Compute Sine - Block ID: 0, Thread ID: 242
// Compute Sine - Block ID: 0, Thread ID: 243
// Compute Sine - Block ID: 0, Thread ID: 244
// Compute Sine - Block ID: 0, Thread ID: 245
// Compute Sine - Block ID: 0, Thread ID: 246
// Compute Sine - Block ID: 0, Thread ID: 247
// Compute Sine - Block ID: 0, Thread ID: 248
// Compute Sine - Block ID: 0, Thread ID: 249
// Compute Sine - Block ID: 0, Thread ID: 250
// Compute Sine - Block ID: 0, Thread ID: 251
// Compute Sine - Block ID: 0, Thread ID: 252
// Compute Sine - Block ID: 0, Thread ID: 253
// Compute Sine - Block ID: 0, Thread ID: 254
// Compute Sine - Block ID: 0, Thread ID: 255
// Compute Sine - Block ID: 0, Thread ID: 128
// Compute Sine - Block ID: 0, Thread ID: 129
// Compute Sine - Block ID: 0, Thread ID: 130
// Compute Sine - Block ID: 0, Thread ID: 131
// Compute Sine - Block ID: 0, Thread ID: 132
// Compute Sine - Block ID: 0, Thread ID: 133
// Compute Sine - Block ID: 0, Thread ID: 134
// Compute Sine - Block ID: 0, Thread ID: 135
// Compute Sine - Block ID: 0, Thread ID: 136
// Compute Sine - Block ID: 0, Thread ID: 137
// Compute Sine - Block ID: 0, Thread ID: 138
// Compute Sine - Block ID: 0, Thread ID: 139
// Compute Sine - Block ID: 0, Thread ID: 140
// Compute Sine - Block ID: 0, Thread ID: 141
// Compute Sine - Block ID: 0, Thread ID: 142
// Compute Sine - Block ID: 0, Thread ID: 143
// Compute Sine - Block ID: 0, Thread ID: 144
// Compute Sine - Block ID: 0, Thread ID: 145
// Compute Sine - Block ID: 0, Thread ID: 146
// Compute Sine - Block ID: 0, Thread ID: 147
// Compute Sine - Block ID: 0, Thread ID: 148
// Compute Sine - Block ID: 0, Thread ID: 149
// Compute Sine - Block ID: 0, Thread ID: 150
// Compute Sine - Block ID: 0, Thread ID: 151
// Compute Sine - Block ID: 0, Thread ID: 152
// Compute Sine - Block ID: 0, Thread ID: 153
// Compute Sine - Block ID: 0, Thread ID: 154
// Compute Sine - Block ID: 0, Thread ID: 155
// Compute Sine - Block ID: 0, Thread ID: 156
// Compute Sine - Block ID: 0, Thread ID: 157
// Compute Sine - Block ID: 0, Thread ID: 158
// Compute Sine - Block ID: 0, Thread ID: 159
// Compute Sine - Block ID: 0, Thread ID: 32
// Compute Sine - Block ID: 0, Thread ID: 33
// Compute Sine - Block ID: 0, Thread ID: 34
// Compute Sine - Block ID: 0, Thread ID: 35
// Compute Sine - Block ID: 0, Thread ID: 36
// Compute Sine - Block ID: 0, Thread ID: 37
// Compute Sine - Block ID: 0, Thread ID: 38
// Compute Sine - Block ID: 0, Thread ID: 39
// Compute Sine - Block ID: 0, Thread ID: 40
// Compute Sine - Block ID: 0, Thread ID: 41
// Compute Sine - Block ID: 0, Thread ID: 42
// Compute Sine - Block ID: 0, Thread ID: 43
// Compute Sine - Block ID: 0, Thread ID: 44
// Compute Sine - Block ID: 0, Thread ID: 45
// Compute Sine - Block ID: 0, Thread ID: 46
// Compute Sine - Block ID: 0, Thread ID: 47
// Compute Sine - Block ID: 0, Thread ID: 48
// Compute Sine - Block ID: 0, Thread ID: 49
// Compute Sine - Block ID: 0, Thread ID: 50
// Compute Sine - Block ID: 0, Thread ID: 51
// Compute Sine - Block ID: 0, Thread ID: 52
// Compute Sine - Block ID: 0, Thread ID: 53
// Compute Sine - Block ID: 0, Thread ID: 54
// Compute Sine - Block ID: 0, Thread ID: 55
// Compute Sine - Block ID: 0, Thread ID: 56
// Compute Sine - Block ID: 0, Thread ID: 57
// Compute Sine - Block ID: 0, Thread ID: 58
// Compute Sine - Block ID: 0, Thread ID: 59
// Compute Sine - Block ID: 0, Thread ID: 60
// Compute Sine - Block ID: 0, Thread ID: 61
// Compute Sine - Block ID: 0, Thread ID: 62
// Compute Sine - Block ID: 0, Thread ID: 63
// Compute Sine - Block ID: 0, Thread ID: 160
// Compute Sine - Block ID: 0, Thread ID: 161
// Compute Sine - Block ID: 0, Thread ID: 162
// Compute Sine - Block ID: 0, Thread ID: 163
// Compute Sine - Block ID: 0, Thread ID: 164
// Compute Sine - Block ID: 0, Thread ID: 165
// Compute Sine - Block ID: 0, Thread ID: 166
// Compute Sine - Block ID: 0, Thread ID: 167
// Compute Sine - Block ID: 0, Thread ID: 168
// Compute Sine - Block ID: 0, Thread ID: 169
// Compute Sine - Block ID: 0, Thread ID: 170
// Compute Sine - Block ID: 0, Thread ID: 171
// Compute Sine - Block ID: 0, Thread ID: 172
// Compute Sine - Block ID: 0, Thread ID: 173
// Compute Sine - Block ID: 0, Thread ID: 174
// Compute Sine - Block ID: 0, Thread ID: 175
// Compute Sine - Block ID: 0, Thread ID: 176
// Compute Sine - Block ID: 0, Thread ID: 177
// Compute Sine - Block ID: 0, Thread ID: 178
// Compute Sine - Block ID: 0, Thread ID: 179
// Compute Sine - Block ID: 0, Thread ID: 180
// Compute Sine - Block ID: 0, Thread ID: 181
// Compute Sine - Block ID: 0, Thread ID: 182
// Compute Sine - Block ID: 0, Thread ID: 183
// Compute Sine - Block ID: 0, Thread ID: 184
// Compute Sine - Block ID: 0, Thread ID: 185
// Compute Sine - Block ID: 0, Thread ID: 186
// Compute Sine - Block ID: 0, Thread ID: 187
// Compute Sine - Block ID: 0, Thread ID: 188
// Compute Sine - Block ID: 0, Thread ID: 189
// Compute Sine - Block ID: 0, Thread ID: 190
// Compute Sine - Block ID: 0, Thread ID: 191
// Angles (radians) and their sine values:
// Angle: 0.00, Sine: 0.0000
// Angle: 0.10, Sine: 0.0998
// Angle: 0.20, Sine: 0.1987