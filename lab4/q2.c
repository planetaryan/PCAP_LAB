#include<stdio.h>
#include "mpi.h"

int main(int argc,char*argv[]){

    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0){
        int matrix[3][3],key,recv_sum,total_sum=0;
        printf("Enter elements\n");
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                scanf("%d",&matrix[i][j]);
            }
        }

        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                printf("%d",matrix[i][j]);
            }
            printf("\n");
        }

        printf("Enter key:");
        scanf("%d",&key);
    MPI_Send(&matrix[0],3,MPI_INT,1,0,MPI_COMM_WORLD);
    MPI_Send(&matrix[1],3,MPI_INT,2,0,MPI_COMM_WORLD);
    MPI_Send(&matrix[2],3,MPI_INT,3,0,MPI_COMM_WORLD);
    MPI_Send(&key,1,MPI_INT,1,1,MPI_COMM_WORLD);
    MPI_Send(&key,1,MPI_INT,2,1,MPI_COMM_WORLD);
    MPI_Send(&key,1,MPI_INT,3,1,MPI_COMM_WORLD);

    MPI_Recv(&recv_sum,1,MPI_INT,1,2,MPI_COMM_WORLD,&status);
    total_sum+=recv_sum;
    MPI_Recv(&recv_sum,1,MPI_INT,2,2,MPI_COMM_WORLD,&status);
    total_sum+=recv_sum;
    MPI_Recv(&recv_sum,1,MPI_INT,3,2,MPI_COMM_WORLD,&status);
    total_sum+=recv_sum;

    printf("Total occurences=%d",total_sum);

    
    }

    else{
        for(int i=1;i<4;i++){
            int recv_arr[3],recv_key,sum=0;
            MPI_Recv(&recv_arr,3,MPI_INT,0,0,MPI_COMM_WORLD,&status);
            MPI_Recv(&recv_key,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);

            for(int j=0;j<3;j++){
                if (recv_arr[j]==recv_key)
                    sum+=1;
            }
            MPI_Send(&sum,1,MPI_INT,0,2,MPI_COMM_WORLD);

        }
    }

    MPI_Finalize();
    return 0;
}