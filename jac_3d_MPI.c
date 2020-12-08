#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "mpi.h"

#define  Max(a, b) ((a) > (b) ? (a) : (b))

double maxeps = 0.1e-7;
unsigned itmax = 100;
double eps, sum;
int N, d_N, wrank, wsize;
double *A;
double *B;
int block_start, block, block_end;

void relax_resid();
void init();
void verify(); 
void update_data();

int main(int argc, char **argv) {
    int b[] = {128, 256, 512, 896, 1024};
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    for (int k = 0; k < 5; k++) {
        N = b[k];
        block = N / wsize;
        block_start = block * wrank;
        block_end = block + block_start;
        d_N = N * N;
        if (block_start != 0) {
            block_start -= 1;
        }
        if (block_end != N) {
            block_end += 1;
        }
        if (!wrank) {
            printf("!!!!!!!      %d       !!!!!!!!!\n", b[k]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        init();
        for(unsigned it = 1; it <= itmax; it++) {
            eps = 0.;
            relax_resid(A, B);
            double *t = A;
            A = B;
            B = t;
            if(eps < maxeps)
                break;
            update_data();
        }
        verify();
        MPI_Barrier(MPI_COMM_WORLD);
        free(A);
        free(B);
        double end = MPI_Wtime();
        if (!wrank)
            printf("SUM: %lf\nTIME: %lf\n", sum, (end - start));
    }
    MPI_Finalize();
    return 0;
}

inline void init() {
    A = malloc(N * d_N * sizeof(double));
    B = malloc(N * d_N * sizeof(double));
    for(unsigned i = block_start; i < block_end; i++)
        for(unsigned j = 0; j <= N - 1; j++)
            for(unsigned k = 0; k <= N - 1; k++) {
                if(i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    *(A + i * d_N + j * N + k) = 0.;
                else
                    *(A + i * d_N + j * N + k) = ( 4. + i + j + k);
            }
} 

inline void relax_resid(double *A, double *B) {
    double t = eps;
#pragma omp parallel for shared (A, B, N, d_N, block_start, block_end) reduction(max:t)
    for(unsigned i = block_start + 1; i < block_end - 1; i++)
        for(unsigned j = 1; j <= N - 2; j++)
            for(unsigned k = 1; k <= N - 2; k++) {
                double e;
                *(B + i * d_N + j * N + k) = (*(A + (i - 1) * d_N + j * N + k) + *(A + (i + 1) * d_N + j * N + k) + *(A + i * d_N + (j - 1) * N + k) 
                        + *(A + i * d_N + (j + 1) * N + k) + *(A + i * d_N + j * N + (k - 1)) + *(A + i * d_N + j * N + (k + 1))) / 6.;
                e = fabs(*(A + i * d_N + j * N + k) - *(B + i * d_N + j * N + k));         
                t = Max(t, e);
            }
    MPI_Allreduce(&t, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

inline void verify() {
    double s = 0.0;
#pragma omp parallel for shared(A, N, d_N) reduction(+:s)
    for(unsigned i = block_start + 1; i < block_end - 1; i++)
        for(unsigned j = 0; j <= N - 1; j++)
            for(unsigned k = 0; k <= N - 1; k++) {
                s += *(A + i * d_N + j * N + k) * (i + 1) * (j + 1) * (k + 1) / (N * d_N);
            }
    MPI_Allreduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

inline void update_data() {
    MPI_Request request[4];
    MPI_Status status[4];
    if (wrank) {
        MPI_Irecv(A + block_start * d_N, d_N, MPI_DOUBLE, wrank - 1, 1215, MPI_COMM_WORLD, &request[0]);
    } if(wrank) {
        MPI_Isend(A + (block_start + 1) * d_N, d_N, MPI_DOUBLE, wrank - 1, 1216, MPI_COMM_WORLD, &request[1]);
    } if(wrank != wsize - 1) {
        MPI_Isend(A + (block_end - 1) * d_N, d_N, MPI_DOUBLE, wrank + 1, 1215, MPI_COMM_WORLD, &request[2]);
    } if (wrank != wsize - 1) {
        MPI_Irecv(A + block_end * d_N, d_N, MPI_DOUBLE, wrank + 1, 1216, MPI_COMM_WORLD, &request[3]);
    }

    int ll = 4, shift = 0; 
    if(!wrank) { 
        ll = 2;
        shift = 2;
    }
    if(wrank == wsize - 1) { 
        ll = 2;
    }
    MPI_Waitall(ll, &request[shift], &status[0]);
}
