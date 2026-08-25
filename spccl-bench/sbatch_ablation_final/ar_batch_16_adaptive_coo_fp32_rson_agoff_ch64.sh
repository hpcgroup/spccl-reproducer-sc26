#!/bin/bash
#SBATCH --job-name ar-16
#SBATCH --nodes 4
#SBATCH --gpus 16
#SBATCH --account m5083_g
#SBATCH --constraint gpu&hbm40g
#SBATCH --time 00:15:00
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
TAG="ar_ablation_backfill"
ITERS=1000
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
# === CCD nccl_sparse runs ===
export NCCL_BUFFSIZE=16777216
export NCCL_CCD_DENSE_THRESHOLD=0.7
export NCCL_CCD_DENSE_INTRA_THRESHOLD=0.8
export NCCL_CCD_AG_DENSE_THRESHOLD=0.1
# Format: adaptive_coo  (NCCL_CCD_FORMAT_MASK=3)
export NCCL_CCD_FORMAT_MASK=3
export NCCL_MIN_NCHANNELS=64
export NCCL_MAX_NCHANNELS=64
srun -N 4 -n 16 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallreduce_fp32 nccl_sparse 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 8192x8192 16384x8192 \
    --sparsities 95.0 \
    --tag "$TAG" --datetime "$DATETIME"

echo "=== Job $JOBID complete ==="
