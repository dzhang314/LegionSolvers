#!/bin/bash -l

module load cuda

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
export GASNET_BACKTRACE=1
export LEGION_BACKTRACE=1

# nvprof -f -o nvprof_${OMPI_COMM_WORLD_RANK}_${OMPI_COMM_WORLD_LOCAL_RANK}.out \
/home/dkzhang/LegionSolvers/build_master_debug/TestExecutable \
    -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 # -lg:warn # -ll:csize 50G -ll:fsize 14G \
#    -lg:prof 12 -lg:prof_logfile prof_%.gz
