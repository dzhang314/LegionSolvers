#!/usr/bin/env python3

import os
import subprocess

from build_utilities import *

count_proc = subprocess.run([
    "jsrun",
    "-r", "1",
    "echo", "hello"
], check=True, capture_output=True)

assert not count_proc.stderr
count_output = count_proc.stdout.splitlines()
for line in count_output:
    assert line == b"hello"

print("Benchmarking on", len(count_output), "nodes.")

NUM_NODES = len(count_output)
NUM_RANKS_PER_NODE = 1
NUM_GPUS_PER_RANK = 4
NUM_CPU_CORES_PER_NODE = 40
NUM_UTIL_CORES_PER_RANK = 4

for PROBLEM_SIZE in [2**i for i in range(10, 40)]:

    remove_directory("legion_prof", quiet=True)
    for i in range(NUM_NODES):
        remove_file(f"prof_{i}.gz", quiet=True)

    proc = subprocess.run([
        "jsrun",
        "--rs_per_host", str(NUM_RANKS_PER_NODE),
        "--cpu_per_rs", str(NUM_CPU_CORES_PER_NODE // NUM_RANKS_PER_NODE),
        "--gpu_per_rs", str(NUM_GPUS_PER_RANK),
        "--bind", "none",
        "/p/gpfs1/zhang70/LegionSolversBuild/cr_gex_cuda_nokokkos_release/Test06CSRSolveCG",
        # "-ll:cpu", str(NUM_CPU_CORES - NUM_UTIL_CORES),
        "-ll:util", str(NUM_UTIL_CORES_PER_RANK),
        "-ll:gpu", str(NUM_GPUS_PER_RANK),
        "-ll:csize", "100G",
        "-ll:fsize", "12G",
        "-lg:eager_alloc_percentage", "20",
        "-n", str(PROBLEM_SIZE),
        "-vp", str(NUM_NODES * NUM_RANKS_PER_NODE * NUM_GPUS_PER_RANK),
        "-it", "1000",
        # "-lg:prof", str(NUM_NODES * NUM_RANKS_PER_NODE),
        # "-lg:prof_logfile", f"prof_CSR_CG_{NUM_NODES}_{PROBLEM_SIZE}_%.gz"
    ], capture_output=True)

    if proc.returncode != 0:
        print("WARNING: PROCESS RETURNED", proc.returncode)

    if proc.stderr:
        print(proc.stderr.decode("utf-8"))

    # print("Generating profiles...")

    # subprocess.run([
    #     "/p/gpfs1/zhang70/lib/legion_master_nocuda_nokokkos_debug/bin/legion_prof.py",
    #     *[f"prof_{i}.gz" for i in range(NUM_NODES)]
    # ], check=True)

    # os.rename("legion_prof", f"prof_{NUM_NODES}_{PROBLEM_SIZE}")

    # print("Finished generating profiles.")

    output = sorted(set(
        line for line in proc.stdout.splitlines()
        if line and b"[LegionSolvers]" not in line
    ))

    try:
        times = [int(line) for line in output]
        diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        diffs = diffs[5:] # discard first few iterations
        print(diffs)
        print(PROBLEM_SIZE, "AVG", sum(diffs) / len(diffs) / 1000000, "ms")
    except:
        print("COULD NOT PARSE STDOUT:", output)
