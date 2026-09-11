#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
llvm_root="${project_root}/3rdparty/llvm-project"
llvm_build_dir="${llvm_root}/build"
llvm_install_dir="${llvm_build_dir}/install"

if [[ ! -f "${llvm_root}/llvm/CMakeLists.txt" ]]; then
    echo "LLVM submodule is missing. Run: git submodule update --init --recursive" >&2
    exit 1
fi

cmake -G Ninja \
    -S "${llvm_root}/llvm" \
    -B "${llvm_build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${llvm_install_dir}" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON

cmake --build "${llvm_build_dir}" --parallel
cmake --install "${llvm_build_dir}"
