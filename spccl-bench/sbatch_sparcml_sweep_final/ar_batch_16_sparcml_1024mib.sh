#!/bin/bash
#SBATCH --job-name ar-16
#SBATCH --nodes 4
#SBATCH --gpus 16
#SBATCH --account m5083_g
#SBATCH --constraint gpu&hbm40g
#SBATCH --time 01:00:00
#SBATCH --qos regular
#SBATCH --exclusive
#SBATCH -o slurm_out/ar_%j.out
#SBATCH -e slurm_out/ar_%j.err

set -o pipefail

mkdir -p slurm_out
mkdir -p results/ar

# --- Minimal NCCL transport tuning (safe for any NCCL version) ---
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn

JOBID=${SLURM_JOB_ID}
DATETIME=$(date +%Y-%m-%d_%H:%M:%S)
N_GPUS=16
TAG="sparcml_final"
ITERS=50
WARMUPS=5
UNIFORM=0

CCD_DIR="$(pwd)"
CSV="${CCD_DIR}/results/ar/ar_${N_GPUS}_${JOBID}.csv"
NCCL_CSV="${CCD_DIR}/results/ar/ar_nccl_${N_GPUS}_${JOBID}.csv"
echo "jobid,method,dtype,n_channels,nccl_ccd_format_mask,iterations,warmups,N,M,val_bytes,dense_bytes,sparsity_pct,uniform_sparsity,total_nnz,min_rank_nnz,max_rank_nnz,avg_rank_nnz,n_gpus,avg_time_no_barrier_no_memcpy,max_time_no_barrier_no_memcpy,min_time_no_barrier_no_memcpy,avg_time_barrier_no_memcpy,max_time_barrier_no_memcpy,min_time_barrier_no_memcpy,avg_time_no_barrier_with_memcpy,max_time_no_barrier_with_memcpy,min_time_no_barrier_with_memcpy,avg_time_barrier_with_memcpy,max_time_barrier_with_memcpy,min_time_barrier_with_memcpy,correct,dense_threshold,dense_intra_threshold,ag_dense_threshold,tag,datetime" > $CSV
echo "jobid,collective,dtype,n_channels,blocking,message_size_mib,n_gpus,time_us_oop,algbw_oop,busbw_oop,time_us_ip,algbw_ip,busbw_ip,tag,datetime" > $NCCL_CSV

# === Load NCCLX (nccl/2.24.3) for CCD runs ===
module load cudatoolkit/12.9
module load nccl/2.24.3

# --- Perlmutter CCD/NCCLX environment (4 GPUs/node) ---
export CUDA_VISIBLE_DEVICES=3,2,1,0
unset SLURM_MPI_TYPE
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"

# CXI/libfabric: force rendezvous mode for all message sizes
export FI_PROVIDER=cxi
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_OPTIMIZED_MRS=0
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1

export NCCL_CROSS_NIC=1
# === SparCML AllReduce runs (CPU-only MPI, fp32) ===
unset NCCL_CCD_FORMAT_MASK
unset NCCL_BUFFSIZE
unset FI_CXI_RDZV_THRESHOLD
unset FI_CXI_RDZV_GET_MIN
unset FI_CXI_RDZV_EAGER_SIZE
unset FI_CXI_OPTIMIZED_MRS
unset FI_CXI_DISABLE_HMEM_DEV_REGISTER
export FI_PROVIDER=cxi
# SparCML: sparcml_recdoubling
srun -N 4 -n 16 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallreduce_sparcml_fp32 sparcml_recdoubling 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x16384 \
    --sparsities 99.9 99.5 99.0 97.0 95.0 90.0 85.0 80.0 70.0 \
    --tag "$TAG" --datetime "$DATETIME"
# SparCML: sparcml_big
srun -N 4 -n 16 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallreduce_sparcml_fp32 sparcml_big 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x16384 \
    --sparsities 99.9 99.5 99.0 97.0 95.0 90.0 85.0 80.0 70.0 \
    --tag "$TAG" --datetime "$DATETIME"
# SparCML: sparcml_small
srun -N 4 -n 16 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallreduce_sparcml_fp32 sparcml_small 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x16384 \
    --sparsities 99.9 99.5 99.0 97.0 95.0 90.0 85.0 80.0 70.0 \
    --tag "$TAG" --datetime "$DATETIME"
# SparCML: sparcml_ring
srun -N 4 -n 16 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallreduce_sparcml_fp32 sparcml_ring 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x16384 \
    --sparsities 99.9 99.5 99.0 97.0 95.0 90.0 85.0 80.0 70.0 \
    --tag "$TAG" --datetime "$DATETIME"

echo "=== Job $JOBID complete ==="
