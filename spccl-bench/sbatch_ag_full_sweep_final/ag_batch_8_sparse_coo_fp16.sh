#!/bin/bash
#SBATCH --job-name ag-8
#SBATCH --nodes 2
#SBATCH --gpus 8
#SBATCH --account m5083_g
#SBATCH --constraint gpu&hbm40g
#SBATCH --time 01:30:00
#SBATCH --qos regular
#SBATCH --exclusive
#SBATCH -o slurm_out/ag_%j.out
#SBATCH -e slurm_out/ag_%j.err

set -o pipefail

mkdir -p slurm_out
mkdir -p results/ag

# --- Minimal NCCL transport tuning (safe for any NCCL version) ---
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn

JOBID=${SLURM_JOB_ID}
DATETIME=$(date +%Y-%m-%d_%H:%M:%S)
N_GPUS=8
TAG="ag_full_sweep_final"
ITERS=1000
WARMUPS=5
UNIFORM=0

CCD_DIR="$(pwd)"
CSV="${CCD_DIR}/results/ag/ag_${N_GPUS}_${JOBID}.csv"
NCCL_CSV="${CCD_DIR}/results/ag/ag_nccl_${N_GPUS}_${JOBID}.csv"
echo "jobid,method,dtype,n_channels,nccl_ccd_format_mask,iterations,warmups,N,M,val_bytes,dense_bytes,sparsity_pct,uniform_sparsity,total_nnz,min_rank_nnz,max_rank_nnz,avg_rank_nnz,n_gpus,avg_time_no_barrier_no_memcpy,max_time_no_barrier_no_memcpy,min_time_no_barrier_no_memcpy,avg_time_barrier_no_memcpy,max_time_barrier_no_memcpy,min_time_barrier_no_memcpy,avg_time_no_barrier_with_memcpy,max_time_no_barrier_with_memcpy,min_time_no_barrier_with_memcpy,avg_time_barrier_with_memcpy,max_time_barrier_with_memcpy,min_time_barrier_with_memcpy,correct,dense_threshold,dense_intra_threshold,ag_dense_threshold,tag,datetime" > $CSV
echo "jobid,collective,dtype,n_channels,blocking,message_size_mib,n_gpus,time_us_oop,algbw_oop,busbw_oop,time_us_ip,algbw_ip,busbw_ip,tag,datetime" > $NCCL_CSV

unset NCCL_CROSS_NIC
# === nccl-tests dense baseline (cudatoolkit/12.9 + nccl/2.24.3) ===
echo "=== Running nccl-tests baseline ==="
module load cudatoolkit/12.9
module load nccl/2.24.3

# --- Perlmutter CCD/NCCLX environment (4 GPUs/node) ---
export CUDA_VISIBLE_DEVICES=3,2,1,0
unset SLURM_MPI_TYPE
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"

# CXI/libfabric: disable rendezvous mode (use eager for all message sizes)
export FI_PROVIDER=cxi
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_OPTIMIZED_MRS=0
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1

unset MPICH_GPU_SUPPORT_ENABLED
cd ../nccl-tests
echo "nccl-tests dir: $(pwd), binary: ./build/all_gather_perf_mpi"
ls -la ./build/all_gather_perf_mpi
unset NCCL_CCD_FORMAT_MASK
# nccl-tests: blocking
unset NCCL_MIN_NCHANNELS
unset NCCL_MAX_NCHANNELS
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 268435456 -e 268435456 -z 1 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=1 -v mib=256.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 536870912 -e 536870912 -z 1 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=1 -v mib=512.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 1073741824 -e 1073741824 -z 1 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=1 -v mib=1024.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 2147483648 -e 2147483648 -z 1 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=1 -v mib=2048.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
# nccl-tests: nonblocking
unset NCCL_MIN_NCHANNELS
unset NCCL_MAX_NCHANNELS
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 268435456 -e 268435456 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=0 -v mib=256.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 536870912 -e 536870912 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=0 -v mib=512.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 1073741824 -e 1073741824 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=0 -v mib=1024.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./build/all_gather_perf_mpi -g 1 -b 2147483648 -e 2147483648 -d half -n $ITERS -w $WARMUPS | \
    tee /dev/stderr | \
    awk -v jobid=$JOBID -v coll=ag -v dt=fp16 -v nch=0 -v blk=0 -v mib=2048.0 -v ng=$N_GPUS -v tag="$TAG" -v dtime="$DATETIME" 'NR>2 && /^[[:space:]]*[0-9]/ {print jobid","coll","dt","nch","blk","mib","ng","$6","$7","$8","$10","$11","$12","tag","dtime}' >> $NCCL_CSV
echo "=== nccl-tests done ==="
cd ${CCD_DIR}

# === Load NCCLX (nccl/2.24.3) for CCD runs ===
module load cudatoolkit/12.9
module load nccl/2.24.3

# --- Perlmutter CCD/NCCLX environment (4 GPUs/node) ---
export CUDA_VISIBLE_DEVICES=3,2,1,0
unset SLURM_MPI_TYPE
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"

# CXI/libfabric: disable rendezvous mode (use eager for all message sizes)
export FI_PROVIDER=cxi
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_OPTIMIZED_MRS=0
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1

export NCCL_CROSS_NIC=1
# === CCD nccl_dense runs (stock NCCLX) ===
unset NCCL_CCD_FORMAT_MASK
unset NCCL_MIN_NCHANNELS
unset NCCL_MAX_NCHANNELS
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallgather_fp16_stock_ncclx nccl_dense 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x8192 16384x16384 32768x16384 32768x32768 \
    --sparsities 0 \
    --tag "$TAG" --datetime "$DATETIME"

# === CCD nccl_sparse dense baseline (sparse path, threshold=1.0, mask=1, sparsity=0) ===
export NCCL_BUFFSIZE=16777216
export NCCL_CCD_FORMAT_MASK=1
export NCCL_CCD_AG_DENSE_THRESHOLD=1.0
export NCCL_CCD_DENSE_THRESHOLD=1.0
export NCCL_CCD_DENSE_INTRA_THRESHOLD=1.0
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallgather_fp16 nccl_sparse 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x8192 16384x16384 32768x16384 32768x32768 \
    --sparsities 0 \
    --tag "$TAG" --datetime "$DATETIME"
export NCCL_MIN_NCHANNELS=64
export NCCL_MAX_NCHANNELS=64
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallgather_fp16 nccl_sparse 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x8192 16384x16384 32768x16384 32768x32768 \
    --sparsities 0 \
    --tag "$TAG" --datetime "$DATETIME"

# === CCD nccl_sparse runs ===
export NCCL_BUFFSIZE=16777216
export NCCL_CCD_AG_DENSE_THRESHOLD=0.0
export NCCL_CCD_DENSE_THRESHOLD=0.0
export NCCL_CCD_DENSE_INTRA_THRESHOLD=0.0
# Format: sparse_coo1d  (NCCL_CCD_FORMAT_MASK=2)
export NCCL_CCD_FORMAT_MASK=2
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=32
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallgather_fp16 nccl_sparse 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x8192 16384x16384 32768x16384 32768x32768 \
    --sparsities 99.9 99.5 99.0 97.0 95.0 90.0 85.0 80.0 70.0 60.0 50.0 40.0 30.0 20.0 10.0 5.0 1.0 \
    --tag "$TAG" --datetime "$DATETIME"
export NCCL_MIN_NCHANNELS=64
export NCCL_MAX_NCHANNELS=64
srun -N 2 -n 8 --ntasks-per-node=4 -c 32 --cpu-bind=cores --mem-bind=local \
    ./bin/test_spallgather_fp16 nccl_sparse 1 $UNIFORM $ITERS $WARMUPS $CSV \
    --sizes 16384x8192 16384x16384 32768x16384 32768x32768 \
    --sparsities 99.9 99.5 99.0 97.0 95.0 90.0 85.0 80.0 70.0 60.0 50.0 40.0 30.0 20.0 10.0 5.0 1.0 \
    --tag "$TAG" --datetime "$DATETIME"

echo "=== Job $JOBID complete ==="
