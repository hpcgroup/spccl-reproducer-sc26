# SpCCL / NCCLX /SparCML Benchmarking Harness

This is a snapshot of the microbenchmarking harness used to evaluate
SpCCL, NCCLX, and SparCML. On Perlmutter, the following will build
the binaries necessary for experiment reproduction.

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

Data to reproduce Figure 2 in the paper (all-gather channel sweep) can be obtained by running
`sbatch_ag_channel_sweep_final/ag_batch_16_sparse_spop_fp32.sh`

Main microbenchmark results (Figures 5-9) can be reproduced by submitting
all batch scripts under: `sbatch_{ar|ag|rs}_full_sweep_final`. These scripts
also produce the data necessary to reproduce Figure 3 (Pici vs COO in SpCCL).
For Figure 9 (all-reduce), SparCML results are included. Thus, scripts under
`sbatch_sparcml_sweep_final` also must be run. 

