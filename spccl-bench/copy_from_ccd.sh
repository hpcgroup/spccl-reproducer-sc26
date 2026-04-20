#!/bin/bash
# Copy minimal CCD files needed to build the RS/AG/AR benchmark drivers
# (including SparCML AllReduce variant) into spccl-bench/.
#
# Run from the spccl-reproducer-sc26 root, or set SRC/DST below.
set -euo pipefail

SRC="${SRC:-../../ccd}"
DST="${DST:-./spccl-bench}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source CCD dir not found at $SRC" >&2
    exit 1
fi

mkdir -p "$DST/include/ccd"
mkdir -p "$DST/examples"
mkdir -p "$DST/external/SparCML"
mkdir -p "$DST/bin"
mkdir -p "$DST/obj"

cp "$SRC/Makefile"                                  "$DST/Makefile"

cp "$SRC/include/ccd/utils.cuh"                     "$DST/include/ccd/"
cp "$SRC/include/ccd/mpi_utils.cuh"                 "$DST/include/ccd/"

cp "$SRC/examples/test_spallgather.cu"              "$DST/examples/"
cp "$SRC/examples/test_spallreduce.cu"              "$DST/examples/"
cp "$SRC/examples/test_spreducescatter.cu"          "$DST/examples/"

cp "$SRC/external/SparCML/c_common.h"               "$DST/external/SparCML/"
cp "$SRC/external/SparCML/c_allreduce_recdoubling.h" "$DST/external/SparCML/"
cp "$SRC/external/SparCML/c_allreduce_big.h"        "$DST/external/SparCML/"
cp "$SRC/external/SparCML/c_allreduce_small.h"      "$DST/external/SparCML/"
cp "$SRC/external/SparCML/c_allreduce_ring.h"       "$DST/external/SparCML/"

# Keep bin/ and obj/ tracked but empty
touch "$DST/bin/.gitkeep" "$DST/obj/.gitkeep"

echo "Copied minimal benchmark sources into $DST"
