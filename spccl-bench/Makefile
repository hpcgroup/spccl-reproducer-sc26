NVCC = nvcc
MPICC = CC
CXXFLAGS = -std=c++20 -Iinclude -arch=sm_80
CXXFLAGS += $(if $(NDEBUG),-DNDEBUG,)
MPICXXFLAGS = -std=c++20 -Iinclude -target-accel=nvidia80 -lcusparse
INCLUDE_DIR = ./include/ccd
EXT_INCLUDE_DIR = ./external
NCCL_DIR = /global/homes/l/ldhough/opt/nccl_2.28.7-1+cuda12.9_x86_64
NCCLX_DIR = ../torchcomms-sparse/build/ncclx
NCCLX_DIR2 = ../torchcomms-sparse/build/ncclx2
ifdef USE_NCCLX2
  NCCLX_DIR = $(NCCLX_DIR2)
endif
ifdef NCCLX_DIR
  NCCL_DIR = $(NCCLX_DIR)
endif
CUDA_DIR = /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux
MPICH_DIR = /opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3
INCLUDE_NCCL = -L$(NCCL_DIR)/lib -I$(NCCL_DIR)/include -lnccl \
			  -Xlinker -rpath -Xlinker '$(NCCL_DIR)/lib'
INCLUDE_CUDA = -L$(CUDA_DIR)/lib64 -I$(CUDA_DIR)/include -lcuda -lcusparse
INCLUDE_MPICH = -L$(MPICH_DIR)/lib -I$(MPICH_DIR)/include

# AllGather: fp32
obj/test_spallgather_fp32.o: examples/test_spallgather.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL) $(INCLUDE_CUDA) $(INCLUDE_MPICH) -c examples/test_spallgather.cu -o $@ -O3

test_spallgather_fp32: obj/test_spallgather_fp32.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL) -O3

# AllGather: fp16
obj/test_spallgather_fp16.o: examples/test_spallgather.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL) $(INCLUDE_CUDA) $(INCLUDE_MPICH) -DUSE_FP16 -c examples/test_spallgather.cu -o $@ -O3

test_spallgather_fp16: obj/test_spallgather_fp16.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL) -O3

# Legacy target name (builds fp32)
test_spallgather: test_spallgather_fp32

# AllReduce: fp32
obj/test_spallreduce_fp32.o: examples/test_spallreduce.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL) $(INCLUDE_CUDA) $(INCLUDE_MPICH) -c examples/test_spallreduce.cu -o $@ -O3

test_spallreduce_fp32: obj/test_spallreduce_fp32.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL) -O3

# AllReduce: fp16
obj/test_spallreduce_fp16.o: examples/test_spallreduce.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL) $(INCLUDE_CUDA) $(INCLUDE_MPICH) -DUSE_FP16 -c examples/test_spallreduce.cu -o $@ -O3

test_spallreduce_fp16: obj/test_spallreduce_fp16.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL) -O3

# Legacy target name (builds fp32)
test_spallreduce: test_spallreduce_fp32

# AllReduce SparCML: fp32
obj/test_spallreduce_sparcml_fp32.o: examples/test_spallreduce.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL) $(INCLUDE_CUDA) $(INCLUDE_MPICH) -I$(EXT_INCLUDE_DIR) -DUSE_SPARCML -Xcompiler -fopenmp -c examples/test_spallreduce.cu -o $@ -O3

test_spallreduce_sparcml_fp32: obj/test_spallreduce_sparcml_fp32.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL) -O3

test_spallreduce_sparcml: test_spallreduce_sparcml_fp32

# ReduceScatter: fp32
obj/test_spreducescatter_fp32.o: examples/test_spreducescatter.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL) $(INCLUDE_CUDA) $(INCLUDE_MPICH) -c examples/test_spreducescatter.cu -o $@ -O3

test_spreducescatter_fp32: obj/test_spreducescatter_fp32.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL) -O3

# ReduceScatter: fp16
obj/test_spreducescatter_fp16.o: examples/test_spreducescatter.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL) $(INCLUDE_CUDA) $(INCLUDE_MPICH) -DUSE_FP16 -c examples/test_spreducescatter.cu -o $@ -O3

test_spreducescatter_fp16: obj/test_spreducescatter_fp16.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL) -O3

# Legacy target name (builds fp32)
test_spreducescatter: test_spreducescatter_fp32

# === stock_ncclx targets: link against torchcomms-baseline (no sparse symbols) ===
STOCK_NCCLX_DIR = ../torchcomms-baseline/build/ncclx
INCLUDE_NCCL_STOCK = -L$(STOCK_NCCLX_DIR)/lib -I$(STOCK_NCCLX_DIR)/include -lnccl \
			  -Xlinker -rpath -Xlinker '$(STOCK_NCCLX_DIR)/lib'
STOCK_NCCLX_DEFS = -DUSE_STOCK_NCCLX

# AllGather stock_ncclx: fp32
obj/test_spallgather_fp32_stock_ncclx.o: examples/test_spallgather.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL_STOCK) $(INCLUDE_CUDA) $(INCLUDE_MPICH) $(STOCK_NCCLX_DEFS) -c examples/test_spallgather.cu -o $@ -O3

test_spallgather_fp32_stock_ncclx: obj/test_spallgather_fp32_stock_ncclx.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL_STOCK) -O3

# AllGather stock_ncclx: fp16
obj/test_spallgather_fp16_stock_ncclx.o: examples/test_spallgather.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL_STOCK) $(INCLUDE_CUDA) $(INCLUDE_MPICH) $(STOCK_NCCLX_DEFS) -DUSE_FP16 -c examples/test_spallgather.cu -o $@ -O3

test_spallgather_fp16_stock_ncclx: obj/test_spallgather_fp16_stock_ncclx.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL_STOCK) -O3

# AllReduce stock_ncclx: fp32
obj/test_spallreduce_fp32_stock_ncclx.o: examples/test_spallreduce.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL_STOCK) $(INCLUDE_CUDA) $(INCLUDE_MPICH) $(STOCK_NCCLX_DEFS) -c examples/test_spallreduce.cu -o $@ -O3

test_spallreduce_fp32_stock_ncclx: obj/test_spallreduce_fp32_stock_ncclx.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL_STOCK) -O3

# AllReduce stock_ncclx: fp16
obj/test_spallreduce_fp16_stock_ncclx.o: examples/test_spallreduce.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL_STOCK) $(INCLUDE_CUDA) $(INCLUDE_MPICH) $(STOCK_NCCLX_DEFS) -DUSE_FP16 -c examples/test_spallreduce.cu -o $@ -O3

test_spallreduce_fp16_stock_ncclx: obj/test_spallreduce_fp16_stock_ncclx.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL_STOCK) -O3

# ReduceScatter stock_ncclx: fp32
obj/test_spreducescatter_fp32_stock_ncclx.o: examples/test_spreducescatter.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL_STOCK) $(INCLUDE_CUDA) $(INCLUDE_MPICH) $(STOCK_NCCLX_DEFS) -c examples/test_spreducescatter.cu -o $@ -O3

test_spreducescatter_fp32_stock_ncclx: obj/test_spreducescatter_fp32_stock_ncclx.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL_STOCK) -O3

# ReduceScatter stock_ncclx: fp16
obj/test_spreducescatter_fp16_stock_ncclx.o: examples/test_spreducescatter.cu $(INCLUDE_DIR)/utils.cuh $(INCLUDE_DIR)/mpi_utils.cuh
	$(NVCC) $(CXXFLAGS) $(INCLUDE_NCCL_STOCK) $(INCLUDE_CUDA) $(INCLUDE_MPICH) $(STOCK_NCCLX_DEFS) -DUSE_FP16 -c examples/test_spreducescatter.cu -o $@ -O3

test_spreducescatter_fp16_stock_ncclx: obj/test_spreducescatter_fp16_stock_ncclx.o
	$(MPICC) $(MPICXXFLAGS) -o bin/$@ $^ $(INCLUDE_NCCL_STOCK) -O3

.PHONY: clean
clean:
	rm -f obj/*.o
	rm -f bin/*

