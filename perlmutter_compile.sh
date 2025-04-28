
cmake -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
    -D Kokkos_ARCH_AMPERE80=ON -D Kokkos_ARCH_ZEN3=ON \
    -D Athena_ENABLE_MPI=ON \
    -D CMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver3" \
    -D PROBLEM=boosted_bh ../

# export ATHENA_DIR=/global/homes/y/ykim/athenak

#    -D CMAKE_CXX_COMPILER=${ATHENA_DIR}/kokkos/bin/nvcc_wrapper \
# -D CMAKE_CXX_FLAGS_RELEASE="-O3 -march=znver3" \
