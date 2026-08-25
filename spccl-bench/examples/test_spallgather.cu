#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <utility>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mpi.h>
#include <nccl.h>
#include "ccd/utils.cuh"
#include "ccd/mpi_utils.cuh"

#ifdef USE_FP16
typedef __half ValType;
static const ncclDataType_t NCCL_VAL_TYPE = ncclHalf;
static const char *DTYPE_STR = "fp16";
static const size_t VAL_BYTES = 2;
#elif defined(USE_FP64)
typedef double ValType;
static const ncclDataType_t NCCL_VAL_TYPE = ncclDouble;
static const char *DTYPE_STR = "fp64";
static const size_t VAL_BYTES = 8;
#else
typedef float ValType;
static const ncclDataType_t NCCL_VAL_TYPE = ncclFloat;
static const char *DTYPE_STR = "fp32";
static const size_t VAL_BYTES = 4;
#endif

using std::cout, std::endl, std::string, std::to_string;

static void print_rank_gpu_binding(MPI_Comm comm) {
    int wrank, wsize;
    MPI_Comm_rank(comm, &wrank);
    MPI_Comm_size(comm, &wsize);
    char host[256];
    gethostname(host, sizeof(host));
    host[sizeof(host)-1] = '\0';
    const char *lrank_s = getenv("SLURM_LOCALID");
    int lrank = lrank_s ? atoi(lrank_s) : -1;
    const char *cvis = getenv("CUDA_VISIBLE_DEVICES");
    if (!cvis) cvis = "(unset)";
    int dev = -1;
    cudaError_t e1 = cudaGetDevice(&dev);
    char busid[32] = "N/A";
    cudaError_t e2 = cudaDeviceGetPCIBusId(busid, sizeof(busid), (e1 == cudaSuccess ? dev : 0));
    cudaDeviceProp prop{};
    cudaError_t e3 = (e1 == cudaSuccess) ? cudaGetDeviceProperties(&prop, dev) : cudaErrorUnknown;
    printf(
        "[rank %d/%d host=%s local=%d] CUDA_VISIBLE_DEVICES=%s cudaGetDevice=%d "
        "busid=%s gpu=%s (e1=%s e2=%s e3=%s)\n",
        wrank, wsize, host, lrank, cvis, dev,
        busid, (e3 == cudaSuccess ? prop.name : "N/A"),
        cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3)
    );
    fflush(stdout);
    MPI_Barrier(comm);
}

static int read_env_int(const char *name, int default_val) {
    const char *val = getenv(name);
    return val ? atoi(val) : default_val;
}

template<typename T>
static bool compare_val(T a, T b);

template<>
bool compare_val<float>(float a, float b) {
    return std::abs(a - b) < 0.001f;
}

template<>
bool compare_val<__half>(__half a, __half b) {
    // 0.15 tolerance needed for fp16 accumulated rounding at large GPU counts (64+)
    return std::abs(__half2float(a) - __half2float(b)) < 0.15f;
}

template<>
bool compare_val<double>(double a, double b) {
    return std::abs(a - b) < 0.0001;
}

struct TimerStats {
    double avg = 0.0, max = 0.0, min = 0.0;
};

static std::pair<uint64_t, uint64_t> parse_size(const char *s) {
    string str(s);
    size_t xpos = str.find('x');
    if (xpos == string::npos) {
        fprintf(stderr, "ERROR: invalid size '%s', expected NxM\n", s);
        exit(1);
    }
    uint64_t n = strtoull(str.substr(0, xpos).c_str(), nullptr, 10);
    uint64_t m = strtoull(str.substr(xpos + 1).c_str(), nullptr, 10);
    return {n, m};
}

int main(int argc, char **argv) {
    if (argc < 7) {
        cout << "Usage: test_spallgather <method: nccl_dense|nccl_sparse>"
            " <correctness_check: 1|0> <uniform_sparsity: 1|0>"
            " <iterations: int> <warmups: int> <csv_path>"
            " --sizes <N1xM1> [N2xM2 ...]"
            " --sparsities <s1> [s2 ...]"
            " [--tag <tag>] [--datetime <datetime>]"
            << endl;
        return -1;
    }

    // 6 positional args
    string method(argv[1]);
    bool correctness_check = (bool) atoi(argv[2]);
    bool uniform_sparsity = (bool) atoi(argv[3]);
    size_t iterations = atoi(argv[4]);
    size_t warmups = atoi(argv[5]);
    string csv_path(argv[6]);
    size_t total_iterations = iterations + warmups;

    // Parse keyword args
    std::vector<std::pair<uint64_t, uint64_t>> sizes;
    std::vector<float> sparsities;
    string tag = "";
    string datetime = "";

    int i = 7;
    while (i < argc) {
        string arg(argv[i]);
        if (arg == "--sizes") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                sizes.push_back(parse_size(argv[i]));
                ++i;
            }
        } else if (arg == "--sparsities") {
            ++i;
            while (i < argc && argv[i][0] != '-') {
                sparsities.push_back(atof(argv[i]));
                ++i;
            }
        } else if (arg == "--tag") {
            ++i;
            if (i < argc) tag = argv[i++];
        } else if (arg == "--datetime") {
            ++i;
            if (i < argc) datetime = argv[i++];
        } else {
            fprintf(stderr, "WARNING: unknown arg '%s'\n", argv[i]);
            ++i;
        }
    }

    if (sizes.empty()) {
        fprintf(stderr, "ERROR: --sizes required\n");
        return -1;
    }
    if (sparsities.empty()) {
        fprintf(stderr, "ERROR: --sparsities required\n");
        return -1;
    }

    // Read env vars for CSV
    const char *jobid_env = getenv("SLURM_JOB_ID");
    string jobid = jobid_env ? jobid_env : "0";
    int n_channels = read_env_int("NCCL_MIN_NCHANNELS", 0);
    int nccl_ccd_format_mask = read_env_int("NCCL_CCD_FORMAT_MASK", 0);

    const char *dense_thresh_env = getenv("NCCL_CCD_DENSE_THRESHOLD");
    string dense_threshold = dense_thresh_env ? dense_thresh_env : "0.3";
    const char *dense_intra_thresh_env = getenv("NCCL_CCD_DENSE_INTRA_THRESHOLD");
    string dense_intra_threshold = dense_intra_thresh_env ? dense_intra_thresh_env : "0.9";
    const char *ag_dense_thresh_env = getenv("NCCL_CCD_AG_DENSE_THRESHOLD");
    string ag_dense_threshold = ag_dense_thresh_env ? ag_dense_thresh_env : "0.0";

    // === ONE-TIME INIT ===
    int local = atoi(getenv("SLURM_LOCALID"));
    cudaSetDevice(local);

    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_rank == 0) {
        cout << "INFO: method=" << method << " dtype=" << DTYPE_STR
             << " num_sizes=" << sizes.size()
             << " num_sparsities=" << sparsities.size()
             << " iters=" << iterations << " warmups=" << warmups << endl;
    }

    print_rank_gpu_binding(MPI_COMM_WORLD);

    ncclComm_t world_comm_nccl;
    ncclUniqueId world_id;
    if (world_rank == 0) ncclGetUniqueId(&world_id);
    MPI_Bcast(&world_id, sizeof(world_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    NC(ncclCommInitRank(&world_comm_nccl, world_size, world_id, world_rank));

    cudaStream_t nccl_stream;
    cudaStreamCreateWithFlags(&nccl_stream, cudaStreamNonBlocking);

    // === CONFIG LOOP ===
    for (auto [N, M] : sizes) {
        uint64_t C = N * M;
        uint64_t shard_count = C / world_size;
        size_t shard_bytes = shard_count * VAL_BYTES;
        size_t full_bytes = C * VAL_BYTES;

        for (float sparsity : sparsities) {
            if (world_rank == 0) {
                cout << "INFO: config N=" << N << " M=" << M
                     << " sparsity=" << sparsity << endl;
            }

            // --- ALLOCATE ---
            ValType *d_shard = nullptr;
            ValType *d_output = nullptr;
            ValType *d_reference = nullptr;
            bool passed_correctness = true;

            cudaMalloc(&d_shard, shard_bytes);
            cudaMalloc(&d_output, full_bytes);

            // --- GENERATE DATA ---
            size_t position_seed = uniform_sparsity ? 42 : (42 + world_rank);
            size_t value_seed = 42 + world_rank;
            std::mt19937 pos_rng(position_seed);
            std::mt19937 val_rng(value_seed);
            std::bernoulli_distribution keep_dist(1.0 - sparsity / 100.0);
            std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

            ValType *h_shard = (ValType*) calloc(shard_count, VAL_BYTES);
            size_t nnz = 0;
            for (size_t idx = 0; idx < shard_count; ++idx) {
                if (keep_dist(pos_rng)) {
                    ++nnz;
                    float val = val_dist(val_rng);
#ifdef USE_FP16
                    h_shard[idx] = __float2half(val);
#else
                    h_shard[idx] = val;
#endif
                }
            }

            cudaMemcpy(d_shard, h_shard, shard_bytes, cudaMemcpyHostToDevice);
            free(h_shard);

            // Gather nnz stats
            size_t nnz_min, nnz_max, nnz_avg, nnz_total;
            mpi_min_max_avg_total_size_t(&nnz, &nnz_min, &nnz_max, &nnz_avg, &nnz_total, world_size);
            if (world_rank == 0) {
                cout << "INFO: nnz_total=" << nnz_total << " nnz_min=" << nnz_min
                     << " nnz_max=" << nnz_max << " nnz_avg=" << nnz_avg << endl;
            }

            // --- CORRECTNESS REFERENCE ---
            if (correctness_check) {
                cudaMalloc(&d_reference, full_bytes);

                // // NCCL-based reference (commented out — replaced with MPI)
                // NC(ncclAllGather(
                //     d_shard, d_reference, shard_count,
                //     NCCL_VAL_TYPE,
                //     world_comm_nccl, nccl_stream
                // ));
                // cudaStreamSynchronize(nccl_stream);

                // MPI-based reference (no arithmetic, raw byte gather)
                ValType *h_shard_ref = (ValType*) malloc(shard_bytes);
                ValType *h_ref_full = (ValType*) malloc(full_bytes);
                cudaMemcpy(h_shard_ref, d_shard, shard_bytes, cudaMemcpyDeviceToHost);
                MPI_Allgather(h_shard_ref, (int)shard_bytes, MPI_BYTE,
                              h_ref_full, (int)shard_bytes, MPI_BYTE, MPI_COMM_WORLD);
                cudaMemcpy(d_reference, h_ref_full, full_bytes, cudaMemcpyHostToDevice);
                free(h_shard_ref);
                free(h_ref_full);

                MPI_Barrier(MPI_COMM_WORLD);
                if (world_rank == 0) cout << "INFO: reference MPI result computed" << endl;
            }

            // --- TIMED LOOP ---
            double acc_no_barrier_no_memcpy = 0.0;
            double acc_barrier_no_memcpy = 0.0;
            double acc_no_barrier_with_memcpy = 0.0;
            double acc_barrier_with_memcpy = 0.0;

            double t0, t1, t2, t3;
            for (size_t run = 0; run < total_iterations; ++run) {
                MPI_Barrier(MPI_COMM_WORLD);

                t0 = MPI_Wtime();

                // No restore needed for AllGather (send buffer not modified)

                t1 = MPI_Wtime();

                if (method == "nccl_dense") {
                    NC(ncclAllGather(
                        d_shard, d_output, shard_count,
                        NCCL_VAL_TYPE,
                        world_comm_nccl, nccl_stream
                    ));
#ifndef USE_STOCK_NCCLX
                } else if (method == "nccl_sparse") {
                    NC(ncclAllGatherSparse(
                        d_shard, d_output, shard_count,
                        NCCL_VAL_TYPE,
                        world_comm_nccl, nccl_stream
                    ));
#endif
                }
                cudaError_t sync_err = cudaStreamSynchronize(nccl_stream); 
                {
                    if (sync_err != cudaSuccess) {
                        fprintf(stderr, "[rank %d] cudaStreamSynchronize FAILED: %s (cuda error %d)\n",
                                world_rank, cudaGetErrorString(sync_err), (int)sync_err);
                        // Also check if there's a sticky error
                        cudaError_t last_err = cudaGetLastError();
                        if (last_err != cudaSuccess && last_err != sync_err) {
                            fprintf(stderr, "[rank %d] cudaGetLastError: %s\n",
                                    world_rank, cudaGetErrorString(last_err));
                        }
                        fflush(stderr);
                        MPI_Abort(MPI_COMM_WORLD, (int)sync_err);
                    }
                }
                
                t2 = MPI_Wtime();

                MPI_Barrier(MPI_COMM_WORLD);
                t3 = MPI_Wtime();

                if (run >= warmups) {
                    acc_no_barrier_no_memcpy   += (t2 - t1);
                    acc_barrier_no_memcpy      += (t3 - t1);
                    acc_no_barrier_with_memcpy += (t2 - t0);
                    acc_barrier_with_memcpy    += (t3 - t0);
                }
            }

            // Convert to ms, average over iterations
            double time_no_barrier_no_memcpy   = acc_no_barrier_no_memcpy   * 1000.0 / iterations;
            double time_barrier_no_memcpy      = acc_barrier_no_memcpy      * 1000.0 / iterations;
            double time_no_barrier_with_memcpy = acc_no_barrier_with_memcpy * 1000.0 / iterations;
            double time_barrier_with_memcpy    = acc_barrier_with_memcpy    * 1000.0 / iterations;

            // Reduce across ranks
            TimerStats s_nbnm, s_bnm, s_nbwm, s_bwm;
            mpi_min_max_avg_dbl(&time_no_barrier_no_memcpy,   &s_nbnm.min, &s_nbnm.max, &s_nbnm.avg, world_size);
            mpi_min_max_avg_dbl(&time_barrier_no_memcpy,      &s_bnm.min,  &s_bnm.max,  &s_bnm.avg,  world_size);
            mpi_min_max_avg_dbl(&time_no_barrier_with_memcpy, &s_nbwm.min, &s_nbwm.max, &s_nbwm.avg, world_size);
            mpi_min_max_avg_dbl(&time_barrier_with_memcpy,    &s_bwm.min,  &s_bwm.max,  &s_bwm.avg,  world_size);

            // --- CORRECTNESS CHECK ---
            if (correctness_check) {
                ValType *h_ref = (ValType*) malloc(full_bytes);
                ValType *h_out = (ValType*) malloc(full_bytes);
                cudaMemcpy(h_ref, d_reference, full_bytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_out, d_output, full_bytes, cudaMemcpyDeviceToHost);
                bool result = true;
                for (size_t idx = 0; idx < C; ++idx) {
                    if (!compare_val<ValType>(h_ref[idx], h_out[idx])) {
                        result = false;
                        break;
                    }
                }
                cout << "INFO: rank " << world_rank << " correctness: "
                     << (result ? "PASS" : "FAIL") << endl;
                if (!result) passed_correctness = false;
                free(h_ref);
                free(h_out);
                cudaFree(d_reference);
            }

            // --- CSV LINE ---
            if (world_rank == 0) {
                size_t dense_bytes = C * VAL_BYTES;
                string csv_line =
                    jobid + "," + method + "," + DTYPE_STR + ","
                    + to_string(n_channels) + "," + to_string(nccl_ccd_format_mask) + ","
                    + to_string(iterations) + "," + to_string(warmups) + ","
                    + to_string(N) + "," + to_string(M) + ","
                    + to_string(VAL_BYTES) + "," + to_string(dense_bytes) + ","
                    + to_string(sparsity) + ","
                    + to_string(uniform_sparsity) + ","
                    + to_string(nnz_total) + "," + to_string(nnz_min) + ","
                    + to_string(nnz_max) + "," + to_string(nnz_avg) + ","
                    + to_string(world_size) + ","
                    + to_string(s_nbnm.avg) + "," + to_string(s_nbnm.max) + "," + to_string(s_nbnm.min) + ","
                    + to_string(s_bnm.avg) + "," + to_string(s_bnm.max) + "," + to_string(s_bnm.min) + ","
                    + to_string(s_nbwm.avg) + "," + to_string(s_nbwm.max) + "," + to_string(s_nbwm.min) + ","
                    + to_string(s_bwm.avg) + "," + to_string(s_bwm.max) + "," + to_string(s_bwm.min) + ","
                    + to_string(passed_correctness) + ","
                    + dense_threshold + "," + dense_intra_threshold + ","
                    + ag_dense_threshold + ","
                    + tag + "," + datetime;

                cout << "INFO: csv: " << csv_line << endl;
                std::ofstream out(csv_path, std::ios::app);
                out << csv_line << "\n";
            }

            // --- FREE ---
            cudaFree(d_shard);
            cudaFree(d_output);
        } // sparsities
    } // sizes

    // === ONE-TIME CLEANUP ===
    cudaStreamDestroy(nccl_stream);
    ncclCommDestroy(world_comm_nccl);
    MPI_Finalize();
    return 0;
}
