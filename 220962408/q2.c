#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int num = 42; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {

        for (int i = 1; i < size; i++) {
            MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            printf("Master (rank %d) sent number %d to rank %d\n", rank, num, i);
        }
    } else {

        MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,&status);
        printf("Slave (rank %d) received number %d\n", rank, num);
    }

    MPI_Finalize();
    return 0;
}
