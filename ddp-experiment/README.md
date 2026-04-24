# Distributed Data Parallel Experiments

Reproduce sparse distributed data parallel experiments on **NERSC Perlmutter**.

## Repositories

Place these repos side-by-side in a directory (e.g. `$SCRATCH/sparsecomms/`):

| Directory | Repo | Branch |
|---|---|---|
| `Megatron-AxoNN/` | [Megatron-AxoNN](https://github.com/axonn-ai/Megatron-AxoNN) | `sparse-collectives` |
| `axonn/` | [axonn](https://github.com/axonn-ai/axonn) | `sparse-collectives` |

## Setup

### 1. Install AxoNN (editable)

```bash
cd axonn
pip install -e .
```

### 2. Prepare the training dataset

A convenience script handles everything — downloading vocab files, fetching a corpus via HuggingFace, and preprocessing it into Megatron's binary format:

```bash
cd $MEGATRON_HOME
bash examples/setup_data.sh              # wikitext-103 (~500 MB, good for sanity checks)
# bash examples/setup_data.sh --dataset pile  # The Pile (much larger, needs HF token)
```

**What the script does, step by step:**

1. **Downloads vocabulary files** (`gpt2-vocab.json` and `gpt2-merges.txt`) into `$DATA_DIR`
2. **Fetches a raw corpus** using the HuggingFace `datasets` library and writes it as JSONL (one `{"text": "..."}` object per line)
3. **Tokenizes and binarizes** the JSONL into Megatron's mmap format using `tools/preprocess_data.py`

After it finishes, `$DATA_DIR` should contain:

```
gpt2-vocab.json
gpt2-merges.txt
raw_corpus.jsonl
BookCorpusDataset_text_document.bin
BookCorpusDataset_text_document.idx
```

> **Note:** The output is named `BookCorpusDataset` by convention regardless of the actual data source. For full details on collecting your own data (Wikipedia, OpenWebText, etc.) and advanced preprocessing options, see the Megatron-AxoNN README.

## Required Environment Variables

Set these before submitting any jobs:

```bash
export MEGATRON_HOME="$SCRATCH/sparsecomms/Megatron-AxoNN"
export TORCHCOMMS_SPARSE_HOME="$SCRATCH/sparsecomms/torchcomms-sparse"
export DATA_DIR="$SCRATCH/sparsecomms/gpt_data"
```

| Variable | Purpose |
|---|---|
| `MEGATRON_HOME` | Root of the Megatron-AxoNN repo |
| `TORCHCOMMS_SPARSE_HOME` | Root of the torchcomms-sparse repo (contains `build/ncclx/`) |
| `DATA_DIR` | Directory with `gpt2-vocab.json`, `gpt2-merges.txt`, and `BookCorpusDataset_text_document.{bin,idx}` |
| `EXPERIMENTS_DIR` | Directory containing `base_job.sh` for the experiment suite. Defaults to `$(pwd)` — set it or `cd` into the experiment directory before submitting. |

> **Note:** SpCCL Python wrappers read `TORCHCOMMS_SPARSE_HOME` at import time to locate the NCCLX library. Override individual paths with `NCCLX_BUILD_DIR` and `NCCLX_LIB_PATH` if needed (see `axonn/axonn/sparse_comms.py` and `axonn/axonn/ncclx_shim.py`).

## Running Experiments

Two experiment suites, differing only in model architecture:

| Suite | Directory | Model | GPUs |
|---|---|---|---|
| 40 GB scaling | `experiments_40gb/` | 1.5B (48L, 1600H, 25 heads) | A100 40 GB |
| 80 GB scaling | `experiments_80gb/` | 3.3B (28L, 3072H, 24 heads) | A100 80 GB |

Each suite runs three configurations at 8 / 16 / 32 / 64 GPUs (2 / 4 / 8 / 16 nodes):

| Config | Description |
|---|---|
| `dense_bd` | Dense all-reduce, with timing breakdown |
| `sparse_noea_bd` | Sparse all-reduce (pruning, no error accumulation), with breakdown |
| `sparse_ea_bd` | Sparse all-reduce (pruning + error accumulation), with breakdown |

Submit with `sbatch`:

```bash
# 40 GB experiments
cd experiments_40gb
sbatch submit_all.sh

# 80 GB experiments
cd experiments_80gb
sbatch submit_all.sh
```

Each `submit_all.sh` requests 16 nodes (64 GPUs) and runs all scales sequentially (8 → 16 → 32 → 64 GPUs). Total wall time is approximately **1.5–2 hours**.
