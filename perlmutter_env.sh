module purge

module load cpe/23.12
module load PrgEnv-gnu
module load cudatoolkit/12.4  # Apr 2025: changed from 12.2 to 12.4
module load craype-accel-nvidia80
module load craype-x86-milan
# module load xpmem   # Apr 2025: Cluster seems to have a problem.
# https://docs.nersc.gov/systems/perlmutter/timeline/#april-16-2025
module load gpu/1.0

# export MPICH_GPU_SUPPORT_ENABLED=0

# export CRAY_ACCEL_TARGET=nvidia80
# export CC=cc
# export CXX=CC
# export FC=ftn
# export CUDACXX=$(which nvcc)
# export CUDAHOSTCXX=CC
