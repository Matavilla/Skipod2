#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <signal.h>

#include "mpi.h"
#include "mpi-ext.h"

#define  Max(a, b) ((a) > (b) ? (a) : (b))

double maxeps = 0.1e-7;
unsigned itmax = 100;
double eps, sum;
const int N = 64;
int d_N, wrank, wsize;
double *A;
double *B;
int block_start, block, block_end;

unsigned NUM_BACKUP_PROC = 0; // from total count
int flag_fault = 0;

MPI_Comm my_comm = MPI_COMM_WORLD;

void relax_resid();
void init();
void verify(); 
void update_data();

void save_data();
void restore_data();
static void verbose_errhandler(MPI_Comm* comm, int* err, ...);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(my_comm, &wrank);
    MPI_Comm_size(my_comm, &wsize);
    wsize -= NUM_BACKUP_PROC;

    d_N = N * N;
    block = N / wsize;
    block_start = block * wrank;
    block_end = block + block_start;
    if (block_start != 0) {
        block_start -= 1;
    }
    if (block_end != N) {
        block_end += 1;
    }
    if (wrank > wsize) {
        block_start = 3;
        block_end = 3;
    }

    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(verbose_errhandler, &errh);
    MPI_Comm_set_errhandler(my_comm, errh);

    MPI_Barrier(my_comm);
    double start = MPI_Wtime();
    init();
    for (unsigned it = 0; it <= itmax; it++) {
        const unsigned CONTROL_POINT = 5;
        if (flag_fault) {
            it -= it % CONTROL_POINT;
            flag_fault = 0;
        }
        if (it % CONTROL_POINT == 0) {
            save_data();
        }

        const unsigned IT_FOR_KILL = 16000;
        const unsigned NUMBER_PROC = wsize - 4;
        if (it == IT_FOR_KILL && wrank == NUMBER_PROC) {
//            raise(SIGKILL);
        }

        eps = 0.;
        relax_resid(A, B);
        double *t = A;
        A = B;
        B = t;
        if (wrank == 0) {
            printf("it: %d eps: %lf\n", it, eps);
        }
        if (eps < maxeps)
            break;
        update_data();
        MPI_Barrier(my_comm);
    }
    verify();
    MPI_Barrier(my_comm);
    free(A);
    free(B);
    double end = MPI_Wtime();

    if (!wrank)
        printf("SUM: %lf\nTIME: %lf\n", sum, (end - start));
    MPI_Finalize();
    return 0;
}

inline void init() {
    A = malloc(N * d_N * sizeof(double));
    B = malloc(N * d_N * sizeof(double));
    for (unsigned i = block_start; i < block_end; i++)
        for (unsigned j = 0; j <= N - 1; j++)
            for (unsigned k = 0; k <= N - 1; k++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    *(A + i * d_N + j * N + k) = 0.;
                else
                    *(A + i * d_N + j * N + k) = ( 4. + i + j + k);
            }
} 

inline void relax_resid(double *A, double *B) {
    double t = eps;
#pragma omp parallel for shared (A, B, N, d_N, block_start, block_end) reduction(max:t)
    for (unsigned i = block_start + 1; i < block_end - 1; i++)
        for (unsigned j = 1; j <= N - 2; j++)
            for (unsigned k = 1; k <= N - 2; k++) {
                double e;
                *(B + i * d_N + j * N + k) = (*(A + (i - 1) * d_N + j * N + k) + *(A + (i + 1) * d_N + j * N + k) + *(A + i * d_N + (j - 1) * N + k) 
                        + *(A + i * d_N + (j + 1) * N + k) + *(A + i * d_N + j * N + (k - 1)) + *(A + i * d_N + j * N + (k + 1))) / 6.;
                e = fabs(*(A + i * d_N + j * N + k) - *(B + i * d_N + j * N + k));         
                t = Max(t, e);
            }
    MPI_Allreduce(&t, &eps, 1, MPI_DOUBLE, MPI_MAX, my_comm);
}

inline void verify() {
    double s = 0.0;
    if (wrank == 0) {
  //      block_start -= 1;
    }
    if (wrank == wsize - 1) {
//        block_end += 1;
    }
#pragma omp parallel for shared(A, N, d_N) reduction(+:s)
    for (unsigned i = block_start + 1; i < block_end - 1; i++)
        for (unsigned j = 0; j <= N - 1; j++)
            for (unsigned k = 0; k <= N - 1; k++) {
                s += *(A + i * d_N + j * N + k) * (i + 1) * (j + 1) * (k + 1) / (N * d_N);
            }
    MPI_Allreduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, my_comm);
    if (wrank == 0) {
    //    block_start += 1;
    }
    if (wrank == wsize - 1) {
    //    block_end -= 1;
    }
}

inline void update_data() {
    if (block_start == block_end) {
        return;
    }
    MPI_Request request[4];
    MPI_Status status[4];
    if (wrank) {
        MPI_Irecv(A + block_start * d_N, d_N, MPI_DOUBLE, wrank - 1, 1215, my_comm, &request[0]);
    } 
    if (wrank) {
        MPI_Isend(A + (block_start + 1) * d_N, d_N, MPI_DOUBLE, wrank - 1, 1216, my_comm, &request[1]);
    } 
    if (wrank != wsize - 1) {
        MPI_Isend(A + (block_end - 2) * d_N, d_N, MPI_DOUBLE, wrank + 1, 1215, my_comm, &request[2]);
    } 
    if (wrank != wsize - 1) {
        MPI_Irecv(A + (block_end - 1) * d_N, d_N, MPI_DOUBLE, wrank + 1, 1216, my_comm, &request[3]);
    }

    int ll = 4, shift = 0; 
    if (!wrank) { 
        ll = 2;
        shift = 2;
    }
    if (wrank == wsize - 1) { 
        ll = 2;
    }
    MPI_Waitall(ll, &request[shift], &status[0]);
}

void save_data() {
    if (block_start == block_end) {
        return;
    }
    char name[10];
    sprintf(name, "data_%d", wrank);
    FILE* f = fopen(name, "wb");
    fwrite(A, sizeof(double), N * d_N, f);
    fclose(f);
}

void restore_data() {
    if (block_start == block_end) {
        return;
    }
    char name[10];
    sprintf(name, "data_%d", wrank);
    FILE* f = fopen(name, "rb");
    fread(A, sizeof(double), N * d_N, f);
    fclose(f);
}

static void verbose_errhandler(MPI_Comm* comm, int* err, ...) {
    int eclass;
    MPI_Error_class(*err, &eclass);
    if (MPIX_ERR_PROC_FAILED != eclass) {
        MPI_Abort(*comm, *err);
    }

    MPI_Group group_c, group_f;
    int nf;
    MPIX_Comm_failure_ack(*comm);
    MPIX_Comm_failure_get_acked(*comm, &group_f);
    MPI_Group_size(group_f, &nf);

    char errstr[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(*err, errstr, &len);
    printf("Rank %d / %d:  Notified of error %s. %d found dead: ( ", wrank, wsize, errstr, nf);

    int* ranks_gf = malloc(nf * sizeof(int));
    int* ranks_gc = malloc(nf * sizeof(int));

    MPI_Comm_group(*comm, &group_c);

    for (unsigned i = 0; i < nf; ++i) {
        ranks_gf[i] = i;
    }

    MPI_Group_translate_ranks(group_f, nf, ranks_gf, group_c, ranks_gc);

    for (unsigned i = 0; i < nf; ++i) {
        printf("%d ", ranks_gc[i]);
    }

    printf(")\n");

    if (nf > NUM_BACKUP_PROC) {
        MPI_Abort(*comm, *err);
    }
    NUM_BACKUP_PROC -= nf;

    MPIX_Comm_shrink(*comm, &my_comm);
    MPI_Comm_rank(my_comm, &wrank);
    MPI_Comm_size(my_comm, &wsize);
    
    wsize -= NUM_BACKUP_PROC;
    block = N / wsize;
    block_start = block * wrank;
    block_end = block + block_start;
    if (block_start != 0) {
        block_start -= 1;
    }
    if (block_end != N) {
        block_end += 1;
    }
    if (wrank > wsize) {
        block_start = 3;
        block_end = 3;
    }
    
    restore_data();

    free(ranks_gc);
    free(ranks_gf);

    flag_fault = 1;
}
