#include "mpi.h"
#include<stdio.h>
#include<string.h>

int main(int argc, char*argv[]){

    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    if(rank==0){
        char str1[100],substr[20];
        printf("Enter string:");
        scanf("%s",str1);
        // printf("The string is %s\n",str1);
        int p_no=1;
        for(int i=0;i<strlen(str1);i+=size){
            strncpy(substr,str1+i,size);
            MPI_Send(&substr,strlen(substr),MPI_CHAR,p_no,0,MPI_COMM_WORLD);
            p_no++;
            printf("Sent %s\n",substr);

        }
        int number_recv,sum=0;
        for(int i=1;i<p_no+1;i++){
        MPI_Recv(&number_recv,1,MPI_INT,i,1,MPI_COMM_WORLD,&status);
        printf("Number received: %d\n",number_recv);
        sum+=number_recv;
        printf("Total number of non-vowels: %d\n",sum);
        }
        
    }
    else{
        char str2[20];
        int non_vowels_count=0;
        MPI_Recv(str2,20,MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
        printf("Received %s\n",str2);
        for(int j=0;j<strlen(str2);j++){
            if(str2[j]!='a'&&str2[j]!='e'&&str2[j]!='i'&&str2[j]!='o'&&str2[j]!='u'){
                non_vowels_count++;
            }
        }
        MPI_Send(&non_vowels_count,1,MPI_INT,0,1,MPI_COMM_WORLD);
    }










    MPI_Finalize();
    return 0;
}

// Enter string:helloandwelcome
// Sent hello
// Sent andwe
// Received hello
// Sent lcome
// Received andwe
// Received lcome
// Number received: 3
// Total number of non-vowels: 3
// Number received: 3
// Total number of non-vowels: 6
// Number received: 3
// Total number of non-vowels: 9