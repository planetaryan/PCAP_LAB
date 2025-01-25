#include "mpi.h"
#include<stdio.h>


int main(int argc,char*argv[]){

    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if(rank==0){
        int arr1[size-1];
        for(int i=0;i<size-1;i++){
            printf("Enter a number:");
            scanf("%d",&arr1[i]);
            MPI_Send(&arr1[i],1,MPI_INT,i+1,0,MPI_COMM_WORLD);
            printf("Sent number to process.\n");
        }

    }

    else{
        int arr2[size-1];
        for(int i=0;i<size-1;i++){
            MPI_Recv(&arr2[i],1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
            // printf("Number Received\n");
            int fact=1;
            for(int j=arr2[i];j>0;j--){
                fact*=j;
            }
            printf("Factorial: %d\n",fact);
            MPI_Send(&fact,1,MPI_INT,0,1,MPI_COMM_WORLD);
            printf("Factorial sent back\n");
        }
    }

    if(rank==0){
        int arr3[size-1];
        int sum=0;
        for(int i=0;i<size-1;i++){
            MPI_Recv(&arr3[i],1,MPI_INT,i+1,1,MPI_COMM_WORLD,&status);
            sum+=arr3[i];
        }

        printf("Sum = %d\n",sum);
    }

    MPI_Finalize();
    return 0;
}

// Enter a number:1
// Sent number to process.
// Enter a number:Factorial: 1
// Factorial sent back
// 2
// Sent number to process.
// Enter a number:Factorial: 2
// Factorial sent back
// 3
// Sent number to process.
// Sum = 9
// Factorial: 6
// Factorial sent back
// 4