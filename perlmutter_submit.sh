#!/bin/bash
#SBATCH -A m4575_g
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -c 32
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.out
#SBATCH -J test
#SBATCH -t 0-00:05:00
#SBATCH --mail-user=ykim7@caltech.edu
#SBATCH --mail-type=ALL

source ~/athenak/perlmutter_env.sh

#
# export MPICH_GPU_SUPPORT_ENABLED=0
# export MPICH_GPU_SUPPORT_ENABLED=1

# OpenMP settings:
# export SLURM_CPU_BIND="cores"
# export OMP_NUM_THREADS=1
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

# run the application:
# - applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1
# - Set the num_tasks (-n ##) to be equal to (number of nodes x 4)
srun -n 4 ./src/athena --kokkos-map-device-id-by=mpi_rank -i ../input
