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
NUM_GPUS_PER_NODE = 4
NUM_CPU_CORES = 40
NUM_UTIL_CORES = 4

for PROBLEM_SIZE in [2**i for i in range(10, 40)]:

    remove_directory("legion_prof", quiet=True)
    for i in range(NUM_NODES):
        remove_file(f"prof_{i}.gz", quiet=True)

    proc = subprocess.run([
        "jsrun",
        "--rs_per_host", "1",
        "--cpu_per_rs", str(NUM_CPU_CORES),
        "--gpu_per_rs", str(NUM_GPUS_PER_NODE),
        "--bind", "none",
        "/p/gpfs1/zhang70/LegionSolversBuild/cr_gex_cuda_nokokkos_release/Test06CSRSolveCG",
        "-ll:cpu", str(NUM_CPU_CORES - NUM_UTIL_CORES),
        "-ll:util", str(NUM_UTIL_CORES),
        "-ll:gpu", str(NUM_GPUS_PER_NODE),
        "-ll:csize", "200G",
        "-ll:fsize", "14G",
        "-lg:eager_alloc_percentage", "20",
        "-n", str(PROBLEM_SIZE),
        "-vp", str(NUM_NODES * NUM_GPUS_PER_NODE),
        "-it", "1000",
        "-lg:prof", str(NUM_NODES),
        "-lg:prof_logfile", f"prof_CSR_CG_{NUM_NODES}_{PROBLEM_SIZE}_%.gz"
    ], capture_output=True)

    if proc.returncode != 0:
        print("WARNING: PROCESS RETURNED", proc.returncode)

    # if proc.stderr:
    #     print(proc.stderr.decode("utf-8"))

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

    for line in output:
        if b" : " not in line:
            print("ERROR: UNEXPECTED STDOUT", line)

    parsed = [
        tuple(map(int, line.split(b" : ")))
        for line in output
        if b" : " in line
    ]

    times = [None] * (max(i for i, t in parsed) + 1)
    for i, t in parsed:
        times[i] = t

    diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    print(diffs)
    print(PROBLEM_SIZE, "AVG", sum(diffs) / len(diffs) / 1000, "ms")
