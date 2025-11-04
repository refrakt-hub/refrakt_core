#!/usr/bin/env fish

function get_cuda_version
    set nvcc_path (which nvcc)
    if test -z "$nvcc_path"
        echo "nvcc not found. Please ensure CUDA is installed and nvcc is in your PATH."
        exit 1
    end
    set version_line (nvcc --version | grep "release")
    set cuda_version (string match -r 'release ([0-9]+\.[0-9]+)' $version_line | string replace -r 'release ' '')
    echo $cuda_version
end

set cuda_version (get_cuda_version)

if string match -r '^12' $cuda_version
    echo "Detected CUDA $cuda_version. Installing cuML and cuDF for CUDA 12..."
    pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.6.*" "cuml-cu12==25.6.*"
else if string match -r '^11\.(4|5|6|7|8)' $cuda_version
    echo "Detected CUDA $cuda_version. Installing cuML for CUDA 11..."
    pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu11==25.6.*"
else
    echo "Unsupported or undetected CUDA version: $cuda_version"
    echo "Please install cuML manually. See https://rapids.ai/start.html for details."
    exit 1
end
