export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_NET="AWS Libfabric"
export FI_PROVIDER=cxi
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_OPTIMIZED_MRS=0
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1
unset MPICH_GPU_SUPPORT_ENABLED

srun -u -C gpu -N 4 -n 16 --ntasks-per-node=4 --exclusive ./build/all_reduce_perf_mpi -g 1 -b 512M -e 512M -z 1 -d float
