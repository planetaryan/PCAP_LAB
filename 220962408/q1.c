#include "mpi.h"
#include<stdio.h>
#include<string.h>
#include <ctype.h>

int main(int argc,char*argv[]){

    int len,rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    char str[100],str_back[100],modified_str[100];
    

    if(rank==0){
        scanf("%s",str);
        printf("Rank %d: Sent string\n", rank);

        len=strlen(str);

        
        MPI_Ssend(str,len+1,MPI_CHAR,1,0,MPI_COMM_WORLD);

    }
    
else{
    MPI_Recv(modified_str,100,MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
    printf("Rank %d received: '%s'\n", rank, modified_str);

    for(int i=0;i<strlen(modified_str);i++){
    if (islower(modified_str[i])){
        modified_str[i]=toupper(modified_str[i]);
    }
    }
    len=strlen(modified_str);
    MPI_Ssend(modified_str,len+1,MPI_CHAR,0,0,MPI_COMM_WORLD);
    printf("Sent modified string to rank 0");
    
}
if(rank==0){

        MPI_Recv(str_back,100,MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
        printf("%s",str_back);

    }

    MPI_Finalize();
    return 0;
}
