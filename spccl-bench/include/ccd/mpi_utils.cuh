#ifndef MPI_UTILS_CUH
#define MPI_UTILS_CUH

#include <iostream>
#include <mpi.h>
#include <nccl.h>

using std::cout, std::endl;

#define NC(code) { nccl_check((code), __FILE__, __LINE__); }

inline void nccl_check(
    ncclResult_t code,
    const char *file, int line, bool abort = true
) {
    if (code != ncclSuccess) {
        fprintf(
            stderr, "NCCL_CHECK: %s %s %d\n",
            ncclGetErrorString(code), file, line
        );
        if (abort) exit(code);
    }
}

// create intra- and inter-node communicators
// intra-node is NCCL, inter-node is MPI...
// inter-node comms connect GPUs with shared
// local ranks
void build_communicators(
    MPI_Comm world_comm,
    MPI_Comm &inter_comm,
    ncclComm_t &intra_comm_nccl,
    ncclComm_t &inter_comm_nccl,
    ncclComm_t &world_comm_nccl,
    size_t &intra_size_out,
    int key = -1
) {
    int world_rank, world_size;
    int intra_rank, intra_size;
    int inter_rank, inter_size;

    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    MPI_Comm intra_comm;
    MPI_Comm_split_type(
        world_comm,
        MPI_COMM_TYPE_SHARED,
        key,
        MPI_INFO_NULL,
        &intra_comm
    );
    MPI_Comm_rank(intra_comm, &intra_rank);
    MPI_Comm_size(intra_comm, &intra_size);
    intra_size_out = intra_size;

    MPI_Comm_split(
        world_comm,
        key,
        world_rank,
        &inter_comm
    );
    MPI_Comm_rank(inter_comm, &inter_rank);
    MPI_Comm_size(inter_comm, &inter_size);

    ncclUniqueId id;
    if (intra_rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, intra_comm);
    NC(ncclCommInitRank(&intra_comm_nccl, intra_size, id, intra_rank));
    MPI_Comm_free(&intra_comm);

    ncclUniqueId inter_id;
    if (inter_rank == 0) {
        ncclGetUniqueId(&inter_id);
    }
    MPI_Bcast(&inter_id, sizeof(inter_id), MPI_BYTE, 0, inter_comm);
    NC(ncclCommInitRank(&inter_comm_nccl, inter_size, inter_id, inter_rank));

    // world-wide NCCL communicator (for nccl_dense baseline)
    ncclUniqueId world_id;
    if (world_rank == 0) {
        ncclGetUniqueId(&world_id);
    }
    MPI_Bcast(&world_id, sizeof(world_id), MPI_BYTE, 0, world_comm);
    NC(ncclCommInitRank(&world_comm_nccl, world_size, world_id, world_rank));

    int local = atoi(getenv("SLURM_LOCALID"));
    bool bad = (intra_rank != local);
    if (!bad) {
        cout << "GROUP CREATION GOOD" << endl;
    } else {
        cout << "GROUP CREATION BAD, MEGA FAIL" << endl;
    }
}

void mpi_avg_dbl(
    double *measurement,
    double *avg,
    size_t n_proc
) {
    MPI_Reduce(
        measurement,
        avg,
        1, MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD
    );
    *avg = *avg / n_proc;
}

void mpi_min_max_avg_dbl(
    double *measurement,
    double *min,
    double *max,
    double *avg,
    size_t n_proc
) {
    MPI_Reduce(
        measurement,
        min,
        1, MPI_DOUBLE, MPI_MIN, 0,
        MPI_COMM_WORLD
    );
    MPI_Reduce(
        measurement,
        max,
        1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD
    );
    MPI_Reduce(
        measurement,
        avg,
        1, MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD
    );
    *avg = *avg / n_proc;
}

void mpi_min_max_avg_total_size_t(
    size_t *measurement,
    size_t *min,
    size_t *max,
    size_t *avg,
    size_t *total,
    size_t n_proc
) {
    MPI_Reduce(
        measurement,
        min,
        1, MPI_DOUBLE, MPI_MIN, 0,
        MPI_COMM_WORLD
    );
    MPI_Reduce(
        measurement,
        max,
        1, MPI_DOUBLE, MPI_MAX, 0,
        MPI_COMM_WORLD
    );
    MPI_Reduce(
        measurement,
        avg,
        1, MPI_DOUBLE, MPI_SUM, 0,
        MPI_COMM_WORLD
    );
    *total = *avg;
    *avg = *avg / n_proc;
}


#endif /* MPI_UTILS_CUH */
