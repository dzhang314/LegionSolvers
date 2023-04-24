#!/usr/bin/env python3

import subprocess
import sys

from build_utilities import *


def count_nodes():
    count_proc = subprocess.run([
        "jsrun", "-r", "1", "echo", "hello"
    ], check=True, capture_output=True)
    assert not count_proc.stderr
    count_output = count_proc.stdout.splitlines()
    for line in count_output:
        assert line == b"hello"
    return len(count_output)


NUM_NODES = count_nodes()
NUM_RANKS_PER_NODE = 1
NUM_GPUS_PER_RANK = 4
NUM_CPU_CORES_PER_NODE = 40
NUM_UTIL_CORES_PER_RANK = 4


IGNORED_STDERR_LINES = [
    "Warning: Overriding spectrum-mpi/rolling-release (module loaded must exactly match)",
    "Warning: Using      spectrum-mpi/2020.08.19 to match app's MPI",
    "Please tell John Gyllenhaal (gyllen@llnl.gov, 4-5485) if this MPI env fix doesn't work",
]


print("Benchmarking on", NUM_NODES, "nodes.", flush=True)

for PROBLEM_SIZE in [2**i for i in range(10, 40)]:

    finished = False
    tries = 0
    while not finished:

        command = [
            "jsrun",
            "--rs_per_host", str(NUM_RANKS_PER_NODE),
            "--cpu_per_rs", str(NUM_CPU_CORES_PER_NODE // NUM_RANKS_PER_NODE),
            "--gpu_per_rs", str(NUM_GPUS_PER_RANK),
            "--bind", "none",
            "/p/gpfs1/zhang70/LegionSolversBuild/cr_cuda_nokokkos_release/BenchmarkStencil",
            # "-ll:cpu", str(NUM_CPU_CORES - NUM_UTIL_CORES),
            "-ll:util", str(NUM_UTIL_CORES_PER_RANK),
            "-ll:gpu", str(NUM_GPUS_PER_RANK),
            "-ll:csize", "100G",
            "-ll:fsize", "12G",
            "-lg:eager_alloc_percentage", "20",
            "-nx", str(PROBLEM_SIZE),
            "-vp", str(NUM_NODES * NUM_RANKS_PER_NODE * NUM_GPUS_PER_RANK),
            "-it", "1000",
            # "-lg:prof", str(NUM_NODES * NUM_RANKS_PER_NODE),
            # "-lg:prof_logfile", f"prof_CSR_CG_{NUM_NODES}_{PROBLEM_SIZE}_%.gz"
        ]

        print("Running command:", ' '.join(command), flush=True)
        proc = subprocess.run(command, capture_output=True)

        if proc.returncode != 0:
            print("WARNING: PROCESS RETURNED", proc.returncode)

        stderr = [
            line for line in proc.stderr.decode("utf-8").splitlines()
            if line.strip() and line.strip() not in IGNORED_STDERR_LINES
        ]

        if any("default_report_failed_instance_creation" in line for line in stderr):
            print(PROBLEM_SIZE, " : ", "OUT OF MEMORY!", flush=True)
            sys.exit(0)

        if stderr:
            print(">>>>>>>>>> BEGIN STDERR <<<<<<<<<<", flush=True)
            for line in stderr:
                print(line, flush=True)
            print(">>>>>>>>>>> END STDERR <<<<<<<<<<<", flush=True)

        stdout = [
            line for line in proc.stdout.decode("utf-8").splitlines()
            if line.strip() and "[LegionSolvers]" not in line
        ]

        for line in stdout:
            if line.startswith("Achieved ") and line.endswith("ms per iteration."):
                print(PROBLEM_SIZE, " : ", line[9:-17], flush=True)
                finished = True
                break
        else:
            tries += 1
            if tries >= 3:
                print("ERROR: Tried three times, no success.", flush=True)
                finished = True
