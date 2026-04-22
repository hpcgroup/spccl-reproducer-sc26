# SpCCL / NCCLX / SparCML Benchmarking Harness

***Reminder***: CCD and SPOP were placeholder names used during development for our
collective library and sparse format, described in the final paper as SpCCL and
Pici. Some materials in the reproducer still refer to CCD and SPOP.

CCD=SpCCL, SPOP=Pici

This is a snapshot of the microbenchmarking harness used to evaluate
SpCCL, NCCLX, and SparCML. On Perlmutter, the following will build
the binaries necessary for experiment reproduction. SpCCL, NCCLX,
and nccl-tests should be built first (instructions in sibling directories
`torchcomms-sparse`, `torchcomms-baseline`, and `nccl-tests`). Building
all harnesses here should take <5 minutes.

```
source modules.sh
bash build.sh
```

Make sure to `module load nccl/2.24.3` before running
benchmarks on a compute node. Slurm submission scripts
are present under `sbatch_*` directories. Results
are emitted to CSV files under
`results/{rs|ar|ag}`. These scripts can be submitted
via `sbatch`, for example:
`sbatch sbatch_rs_full_sweep_final/rs_batch_16_adaptive_spop_fp32.sh`.
The account name (after `#SBATCH --account` in the scripts) will need
to be changed to an account the user has access to.

Data to reproduce Figure 2 in the paper (reduce-scatter threshold sweep) can be
obtained by running `sbatch_rs_threshold_sweep_final/rs_batch_16_adaptive_spop_fp32.sh`.

Data to reproduce Figure 4 in the paper (all-gather channel sweep) can be obtained by running
`sbatch_ag_channel_sweep_final/ag_batch_16_sparse_spop_fp32.sh`

Main microbenchmark results (Figures 5-9) can be reproduced by submitting
all batch scripts under: `sbatch_{ar|ag|rs}_full_sweep_final`. These scripts
also produce the data necessary to reproduce Figure 3 (Pici vs COO in SpCCL).
For Figure 9 (all-reduce), SparCML results are included. Thus, scripts under
`sbatch_sparcml_sweep_final` also must be run.

# Results Interpretation

## CSV Field Reference

Each batch script writes two CSV files under `results/{rs|ar|ag}/`:
a CCD (SpCCL)/NCCLX results file (`<coll>_<ngpus>_<jobid>.csv`) and a
nccl-tests baseline file (`<coll>_nccl_<ngpus>_<jobid>.csv`). The
schemas differ. The meaning of fields relevant to reproduction
are enumerated below.

### CCD (SpCCL) / NCCLX CSV (`<coll>_<ngpus>_<jobid>.csv`)

**Run identification**
- `jobid` — SLURM job ID
- `tag` — sweep label set in the batch script (e.g. `rs_full_sweep_final`)
- `method` — implementation/format used (`nccl_dense` for NCCLX, `nccl_sparse` for SpCCL)
    - For SparCML runs, this will be one of: `sparcml_recdoubling`, `sparcml_big`, `sparcml_small`, `sparcml_ring`
- `dtype` — element type (`fp32`)
- `n_gpus` — total GPUs in the run

**Workload**
- `N`, `M` — tensor dimensions; total elements `C = N*M`
- `val_bytes` — bytes per value (4 for fp32, 2 for fp16)
- `dense_bytes` — total dense tensor size (`C * val_bytes`)
- `sparsity_pct` — target sparsity in percent (0 = dense, 99 = 99% zeros)
- `uniform_sparsity` — `1` if every rank has identical sparsity patterns; `0` if sparsity is randomized per rank

**CCD (SpCCL)/NCCLX configuration**
- `n_channels` — value of `NCCL_MIN/MAX_NCHANNELS` for the run (`0` if unset). These are set to the same value in
microbenchmarks to fix a channel count.
- `nccl_ccd_format_mask` — value of `NCCL_CCD_FORMAT_MASK` (1=dense-only, 2=COO non-adaptive, 3=COO adaptive, 4=SPOP/Pici non-adaptive, 5=SPOP/Pici adaptive). "Adaptive" means that the collective may swap data to a dense format during collective execution (relevant to reduce-scatter and all-reduce).
- `dense_threshold` — `NCCL_CCD_DENSE_THRESHOLD` (inter-node sparsity threshold above which the sparse path is used)
- `dense_intra_threshold` — `NCCL_CCD_DENSE_INTRA_THRESHOLD` (intra-node sparsity threshold)
- `ag_dense_threshold` — `NCCL_CCD_AG_DENSE_THRESHOLD` (sparsity threshold for the all-gather phase of RS+AG all-reduce)

**Timing** (all in milliseconds, averaged over `iterations` after `warmups`)

Four timing variants are reported as `{avg_time,max_time,min_time}` reduced across ranks. `avg_time_barrier_no_memcpy` is the timer used in paper plots. It includes a timed barrier after the collective invocation.

**Iteration counts**
- `iterations`, `warmups` — number of timed iterations and discarded warmups

**Validation**
- `correct` — `1` if the GPU result matches the CPU MPI reference checks.

### nccl-tests CSV (`<coll>_nccl_<ngpus>_<jobid>.csv`)

Parsed from nccl-tests stdout, one row per `(message_size, blocking_mode)`:

- `jobid`, `tag`, `n_gpus`, `dtype` — same meaning as above
- `collective` — `ag`, `rs`, or `ar`
- `n_channels` — `NCCL_MIN/MAX_NCHANNELS` value (`0` if unset)
- `blocking` — `1` for blocking API (`-z 1`), `0` for nonblocking. Paper plots report blocking time.
- `message_size_mib` — message size in MiB
- `time_us_oop`, — out-of-place execution time (μs), which is the timer used for NCCL results in the paper.
