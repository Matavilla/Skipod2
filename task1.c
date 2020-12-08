#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int wrank, wsize;
    unsigned L = 8000; // число double в сообщении
    unsigned K = 100; // число кадров в 1 пути для передачи сообщения
    unsigned Count = L / (2 * K); // чисел в кадре
    double* l_array = malloc(L * sizeof(double));
    int paths[2][7] = {{0, 1, 2, 3, 7, 11, 15}, {0, 4, 8, 12, 13, 14, 15}};
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    if (wrank == 0) {
        for (unsigned i = 0; i < L; i++) {
            l_array[i] = i;
        }
    }
    
    for (unsigned i = 0, tag = 1215; i < 2; i++, tag++) {
        for (unsigned j = 1; j < 7; j++) {
            if (paths[i][j] == wrank) {
                unsigned c = 0;
                if (wrank == 15 && i == 1) {
                    c = K * Count;
                }
                for (unsigned k = 0; k < K; k++) {
                    MPI_Recv(l_array + k * Count + c, Count, MPI_DOUBLE, paths[i][j - 1], tag, MPI_COMM_WORLD, &status);
                }
                break;
            }
        }
    }

    unsigned buff_size = L * sizeof(double) + MPI_BSEND_OVERHEAD;
    double* buff = malloc(buff_size);
    MPI_Buffer_attach(buff, buff_size);
    for (unsigned i = 0, tag = 1215; i < 2; i++, tag++) {
        for (unsigned j = 0; j < 6; j++) {
            if (paths[i][j] == wrank) {
                unsigned c = 0;
                if (wrank == 0 && i == 1) {
                    c = K * Count;
                }
                for (unsigned k = 0; k < K; k++) {
                    MPI_Bsend(l_array + k * Count + c, Count, MPI_DOUBLE, paths[i][j + 1], tag, MPI_COMM_WORLD);
                }
                break;
            }
        }
    }
    MPI_Buffer_detach(buff, &buff_size);
    free(buff);
    
    if (wrank == 15) {
        for (unsigned i = 0; i < L; i++) {
            printf("%lf\n", l_array[i]);
        }
    }
    free(l_array);

    MPI_Finalize();
    return 0;
}
