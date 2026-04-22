# SpCCL REPRODUCER

This repository is designed to aid in reproducing results
from the research paper describing SpCCL and Pici. SpCCL
is a sparse communication library forked from NCCLX. It
supports all-reduce, all-gather, and reduce-scatter. Pici
is a new bitvector-based format designed for low space overhead
at moderate sparsities, as well as efficient compression and
decompression on NVIDIA GPUs.

***NOTE:*** The placeholder names "CCD" and "SPOP" were
used in development in place of the final names "SpCCL"
and "Pici". There are still numerous references to CCD
and SPOP in this artifact, so note that CCD is interchangeable
with SpCCL, and SPOP is interchangeable with Pici.

## Reproducer Structure

1. `torchcomms-sparse`: The implementation of SpCCL and associated build scripts.
2. `torchcomms-baseline`: An unmodified (except build scripts) fork of NCCLX used to compare SpCCL to dense NCCLX collectives.
3. `nccl-tests`: A snapshot of NVIDIA's nccl-tests (https://github.com/NVIDIA/nccl-tests) microbenchmarking software.
4. `spccl-bench`: Our microbenchmarking software for SpCCL, NCCLX, and SparCML, including build scripts.
5. `ddp-experiment`: Code to replicate our end-to-end DDP training experiment using SpCCL.

## Build Instructions

All paper experiments were carried out on NERSC's Perlmutter, and reproducibility has only been verified on Perlmutter. However, we consider it likely that the software will build on any similar system with NVIDIA Ampere or later GPUs. Build/installation instructions for each artifact component are provided in their respective directories. Before running microbenchmarks through batch scripts in `spccl-bench`, each of the following should be built:

1. `torchcomms-sparse` (SpCCL)
2. `torchcomms-baseline` (NCCLX)
3. `nccl-tests`

`spccl-bench` microbenchmark harnesses may then be built. `torchcomms-sparse` may take as long as two hours to build on Perlmutter login nodes. `torchcomms-baseline` can reuse some third party dependencies, so the build may be somewhat faster. `nccl-tests` and `spccl-bench` microbenchmarking harnesses build in less than five minutes.

## Execution Instructions

Microbenchmark execution instructions are available in `spccl-bench`.
