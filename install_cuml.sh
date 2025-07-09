#!/bin/bash

set -e
CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]*\.[0-9]*" | awk '{print $2}' | head -n1)
if [[ $CUDA_VERSION == 12* ]]; then
    pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.6.*" "cuml-cu12==25.6.*"
elif [[ $CUDA_VERSION == 11.4* || $CUDA_VERSION == 11.5* || $CUDA_VERSION == 11.6* || $CUDA_VERSION == 11.7* || $CUDA_VERSION == 11.8* ]]; then
    pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu11==25.6.*"
else
    echo "Unsupported CUDA version: $CUDA_VERSION"
    exit 1
fi