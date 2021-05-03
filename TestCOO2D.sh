#!/bin/bash -l

module load cuda

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

nvprof -f -o nvprof2_${OMPI_COMM_WORLD_RANK}_${OMPI_COMM_WORLD_LOCAL_RANK}.out \
    /home/dkzhang/LegionSolvers/build/LegionSolversTestCOO2D \
    -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 -ll:csize 50G -ll:fsize 15G
    #-lg:prof 16 -lg:prof_logfile prof_%.gz
