#!/bin/bash -l

module load cuda

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
export GASNET_BACKTRACE=1
export LEGION_BACKTRACE=1

/home/dkzhang/LegionSolvers/build_master_debug/Test01VectorArithmetic \
    -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 -lg:warn -lg:leaks
