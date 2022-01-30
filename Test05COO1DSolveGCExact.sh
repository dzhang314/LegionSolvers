#!/bin/bash -l

module load cuda

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
export GASNET_BACKTRACE=1
export LEGION_BACKTRACE=1

# /home/dkzhang/LegionSolvers/build_master_debug/Test05COO1DSolveCGExact \
#     -lg:warn -lg:leaks -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1

/home/dkzhang/LegionSolvers/build_master_release/Test05COO1DSolveCGExact \
    -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 -n 400000000 -vp 16 -it 20 \
    -ll:csize 50G -ll:fsize 14G -lg:prof 16 -lg:prof_logfile prof_%.gz
