#!/bin/bash
set -e
make test_spallgather_fp32                 NDEBUG=1
make test_spreducescatter_fp32             NDEBUG=1
make test_spallreduce_fp32                 NDEBUG=1
make test_spallreduce_sparcml_fp32         NDEBUG=1
make test_spallgather_fp32_stock_ncclx     NDEBUG=1
make test_spreducescatter_fp32_stock_ncclx NDEBUG=1
make test_spallreduce_fp32_stock_ncclx     NDEBUG=1
