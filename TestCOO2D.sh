#!/bin/bash -l

module load cuda

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK

# nvprof -f -o nvprof_${OMPI_COMM_WORLD_RANK}_${OMPI_COMM_WORLD_LOCAL_RANK}.out \
/home/dkzhang/LegionSolvers/build_master/TestExecutable \
    -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 -ll:csize 50G -ll:fsize 14G \
    -kp 16 -ip 8 -op 8 -h 1024 -w 1024 -it 64 \
    # -lg:prof 8 -lg:prof_logfile prof_%.gz

# ./LegionSolversTestCOO2D -ll:ocpu 1 -ll:onuma 0 -ll:gpu 1 -ll:csize 50G -ll:fsize 14G -kp 16 -ip 8 -op 8 -h 1024 -w 1024 -it 64
