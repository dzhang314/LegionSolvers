#!/bin/bash -l

module load cuda

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
export GASNET_BACKTRACE=1
export LEGION_BACKTRACE=1

# -gex:bindhost 0

# /home/dkzhang/LegionSolvers/build_master_debug/Test02CGLaplacian1D \
#     -lg:warn -lg:leaks -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1

# /home/dkzhang/LegionSolvers/build_master_gex_release/Test02CGLaplacian1D \
#     -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 -n 16 -vp 1 -kp 1 -it 20 -lg:inorder

/home/dkzhang/LegionSolvers/build_master_release/Test02CGLaplacian1D \
    -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 -n 400000000 -vp 16 -kp 48 -it 20 \
    -ll:csize 50G -ll:fsize 14G -lg:prof 16 -lg:prof_logfile prof_%.gz
