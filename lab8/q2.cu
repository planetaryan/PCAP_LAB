#include<stdio.h>
#include<cuda_runtime.h>

#define N 3

__global__ void matrixByRow(int*A,int*B,int*C,int ha,int wb,int wa){
    int row=threadIdx.x;
    int sum;
    if(row<ha){
        for(int j=0;j<wb;j++){
            sum=0;
            for(int k=0;k<wa;k++){
                sum+=A[row*wa + k]*B[k*wb + j];
            }
            C[row*wb + j]=sum;
        }
        
    }

}

__global__ void matrixByCol(int*A,int*B,int*C,int ha,int wb,int wa){
    int col=threadIdx.x;
    int sum;
    if(col<wb){
        for(int i=0;i<ha;i++){
            sum=0;
            for(int k=0;k<wa;k++){
                sum+=A[i*wa + k]*B[k*wb + col];
            }
            C[i*wb + col]=sum;
        }
        
    }

}

__global__ void matrixByElement(int*A,int*B,int*C,int ha,int wb,int wa){
    int col=threadIdx.x;
    int row=threadIdx.y;
    int sum=0;
    if(col<wb && row<wa){

            for(int k=0;k<wa;k++){
                sum+=A[row*wa + k]*B[k*wb + col];
            }
            C[row*wb + col]=sum;
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
        h_B[i] = i+1 ;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Case (a): Each row of the resultant matrix is computed by one thread
    dim3 blockDim(N, N);
    matrixByElement<<<1, blockDim>>>(d_A, d_B, d_C, N,N,N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Matrix multiplication (row by row):\n");
    printMatrix(h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
