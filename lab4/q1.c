#include<stdio.h>
#include "mpi.h"

int main(int argc, char*argv[]){
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Status status;

    if(rank==0){
        int sum=0;
    for(int i=1;i<size;i++){
        MPI_Send(&i,1,MPI_INT,i,0,MPI_COMM_WORLD);
        printf("Sent %d to process %d\n",i,i);
    }

    int recv_fact;

    for(int k=1;k<size;k++){
            MPI_Recv(&recv_fact,1,MPI_INT,k,1,MPI_COMM_WORLD,&status);
            printf("Received factorial %d from process %d\n",recv_fact,k);
            sum+=recv_fact;
    }
    printf("Sum = %d\n",sum);
    }

    int recv_int,fact=1,sum=0;

    MPI_Recv(&recv_int,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
    printf("Received %d from process 0\n",recv_int);
    for(int j=recv_int;j>0;j--){
        fact*=j;
    }
    MPI_Send(&fact,1,MPI_INT,0,1,MPI_COMM_WORLD);
    printf("Sent factorial %d to process 0\n",fact);

    
    MPI_Finalize();
    return 0;

}
