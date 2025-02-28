#include<stdio.h>
#include<cuda_runtime.h>

__global__ void add(int*a,int*b,int*c){
    int index=threadIdx.x;
    c[index]=a[index]+b[index];
}

int main(){
    int n=5;
    int size=n*sizeof(int);

    int h_a[]={1,2,3,4,5};
    int h_b[]={10,20,30,40,50};
    int h_c[n];

    int*d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    add<<<1,n>>>(d_a,d_b,d_c);

    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    printf("Result: ");
    for(int i=0;i<n;i++){
        printf("%d " ,h_c[i]);
    }
    printf("\n");

    return 0;
}