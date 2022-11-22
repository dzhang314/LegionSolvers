#!/usr/bin/env python3

import os
import subprocess

from build_utilities import *
from make_all import DESIRED_VARIANTS


################################################################################


IGNORED_STDERR_LINES = [
    b"Warning: Overriding spectrum-mpi/rolling-release (module loaded must exactly match)",
    b"Warning: Using      spectrum-mpi/2020.08.19 to match app's MPI",
    b"Please tell John Gyllenhaal (gyllen@llnl.gov, 4-5485) if this MPI env fix doesn't work"
]


def test_00(use_cuda, use_kokkos):
    proc = subprocess.run(["Test00Build"], check=True, capture_output=True)
    assert not proc.stderr
    stdout = proc.stdout.splitlines()
    assert (b"CUDA enabled" if use_cuda else b"CUDA disabled") in stdout
    assert (b"Kokkos enabled" if use_kokkos else b"Kokkos disabled") in stdout
    print("TEST 00 PASSED")


def test_01(use_cuda, use_kokkos):
    command = [
        "jsrun",
        "--rs_per_host", "4" if use_kokkos else "1",
        "--cpu_per_rs", "10" if use_kokkos else "40",
        "--gpu_per_rs", "1" if use_kokkos else "4",
        "--bind", "none",
        "Test01ScalarOperations",
        "-lg:warn", "-lg:leaks",
    ]
    if use_kokkos:
        command += [
            "-ll:ocpu", "1",
            "-ll:onuma", "0",
            "-ll:gpu", "1",
        ]
    proc = subprocess.run(command, capture_output=True)
    stdout = [
        line for line in proc.stdout.splitlines()
        if not (line.startswith(b"[LegionSolvers] Registering task") or
                line.startswith(b"[LegionSolvers] Constructing") or
                line.startswith(b"[LegionSolvers] Finished constructing"))
    ]
    stderr = [
        line for line in proc.stderr.splitlines()
        if not line in IGNORED_STDERR_LINES
    ]
    # assert not stdout
    assert not stderr
    print("TEST 01 PASSED" if proc.returncode == 0 else "TEST 01 FAILED")


def test_02(use_cuda, use_kokkos):
    command = [
        "jsrun",
        "--rs_per_host", "4" if use_kokkos else "1",
        "--cpu_per_rs", "10" if use_kokkos else "40",
        "--gpu_per_rs", "1" if use_kokkos else "4",
        "--bind", "none",
        "Test02VectorOperations",
        "-lg:warn", "-lg:leaks",
    ]
    if use_kokkos:
        command += [
            "-ll:ocpu", "1",
            "-ll:onuma", "0",
            "-ll:gpu", "1",
        ]
    proc = subprocess.run(command, capture_output=True)
    stdout = [
        line for line in proc.stdout.splitlines()
        if not (line.startswith(b"[LegionSolvers] Registering task") or
                line.startswith(b"[LegionSolvers] Constructing") or
                line.startswith(b"[LegionSolvers] Finished constructing"))
    ]
    stderr = [
        line for line in proc.stderr.splitlines()
        if not line in IGNORED_STDERR_LINES
    ]
    assert stdout == [b'0', b'0', b'0', b'0', b'0', b'0']
    assert not stderr
    print("TEST 02 PASSED" if proc.returncode == 0 else "TEST 02 FAILED")


def test_03(use_cuda, use_kokkos):
    command = [
        "jsrun",
        "--rs_per_host", "4" if use_kokkos else "1",
        "--cpu_per_rs", "10" if use_kokkos else "40",
        "--gpu_per_rs", "1" if use_kokkos else "4",
        "--bind", "none",
        "Test03COOPartitioning",
        "-lg:warn", "-lg:leaks",
    ]
    if use_kokkos:
        command += [
            "-ll:ocpu", "1",
            "-ll:onuma", "0",
            "-ll:gpu", "1",
        ]
    proc = subprocess.run(command, capture_output=True)
    stdout = [
        line for line in proc.stdout.splitlines()
        if not (line.startswith(b"[LegionSolvers] Registering task") or
                line.startswith(b"[LegionSolvers] Constructing") or
                line.startswith(b"[LegionSolvers] Finished constructing"))
    ]
    stderr = [
        line for line in proc.stderr.splitlines()
        if not line in IGNORED_STDERR_LINES
    ]
    # print(stdout)
    assert not stderr
    print("TEST 03 PASSED" if proc.returncode == 0 else "TEST 03 FAILED")


def test_04(use_cuda, use_kokkos):
    command = [
        "jsrun",
        "--rs_per_host", "4" if use_kokkos else "1",
        "--cpu_per_rs", "10" if use_kokkos else "40",
        "--gpu_per_rs", "1" if use_kokkos else "4",
        "--bind", "none",
        "Test04CSRPartitioning",
        "-lg:warn", "-lg:leaks",
    ]
    if use_kokkos:
        command += [
            "-ll:ocpu", "1",
            "-ll:onuma", "0",
            "-ll:gpu", "1",
        ]
    proc = subprocess.run(command, capture_output=True)
    stdout = [
        line for line in proc.stdout.splitlines()
        if not (line.startswith(b"[LegionSolvers] Registering task") or
                line.startswith(b"[LegionSolvers] Constructing") or
                line.startswith(b"[LegionSolvers] Finished constructing"))
    ]
    stderr = [
        line for line in proc.stderr.splitlines()
        if not line in IGNORED_STDERR_LINES
    ]
    print(stdout)
    assert not stderr
    print("TEST 04 PASSED" if proc.returncode == 0 else "TEST 04 FAILED")


def test_05(use_cuda, use_kokkos):
    command = [
        "jsrun",
        "--rs_per_host", "4" if use_kokkos else "1",
        "--cpu_per_rs", "10" if use_kokkos else "40",
        "--gpu_per_rs", "1" if use_kokkos else "4",
        "--bind", "none",
        "Test05COOSolveCG",
        "-lg:warn", "-lg:leaks",
    ]
    if use_kokkos:
        command += [
            "-ll:ocpu", "1",
            "-ll:onuma", "0",
            "-ll:gpu", "1",
        ]
    proc = subprocess.run(command, capture_output=True)
    stdout = [
        line for line in proc.stdout.splitlines()
        if not (line.startswith(b"[LegionSolvers] Registering task") or
                line.startswith(b"[LegionSolvers] Constructing") or
                line.startswith(b"[LegionSolvers] Finished constructing"))
    ]
    stderr = [
        line for line in proc.stderr.splitlines()
        if not line in IGNORED_STDERR_LINES
    ]
    assert sorted(stdout) == [b'100', b'3280', b'3444', b'3612',
                              b'3784', b'3960', b'4140', b'4324',
                              b'4512', b'4704', b'4900']
    assert not stderr
    print("TEST 05 PASSED" if proc.returncode == 0 else "TEST 05 FAILED")


def test_06(use_cuda, use_kokkos):
    command = [
        "jsrun",
        "--rs_per_host", "4" if use_kokkos else "1",
        "--cpu_per_rs", "10" if use_kokkos else "40",
        "--gpu_per_rs", "1" if use_kokkos else "4",
        "--bind", "none",
        "Test06CSRSolveCG",
        "-lg:warn", "-lg:leaks",
    ]
    if use_kokkos:
        command += [
            "-ll:ocpu", "1",
            "-ll:onuma", "0",
            "-ll:gpu", "1",
        ]
    proc = subprocess.run(command, capture_output=True)
    stdout = [
        line for line in proc.stdout.splitlines()
        if not (line.startswith(b"[LegionSolvers] Registering task") or
                line.startswith(b"[LegionSolvers] Constructing") or
                line.startswith(b"[LegionSolvers] Finished constructing"))
    ]
    stderr = [
        line for line in proc.stderr.splitlines()
        if not line in IGNORED_STDERR_LINES
    ]
    assert sorted(stdout) == [b'100', b'3280', b'3444', b'3612',
                              b'3784', b'3960', b'4140', b'4324',
                              b'4512', b'4704', b'4900']
    assert not stderr
    print("TEST 06 PASSED" if proc.returncode == 0 else "TEST 06 FAILED")


def main():
    for branch_tag, _ in LEGION_BRANCHES:
        for build_tag, _ in BUILD_TYPES:
            for network_tag, _ in NETWORK_TYPES:
                for cuda_tag, use_cuda in CUDA_TYPES:
                    for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                        build_name = underscore_join(
                            branch_tag, network_tag,
                            cuda_tag, kokkos_tag, build_tag
                        )
                        if build_name not in DESIRED_VARIANTS:
                            continue
                        build_dir = os.path.join(
                            SCRATCH_DIR,
                            "LegionSolversBuild",
                            build_name
                        )
                        with change_directory(build_dir):
                            test_00(use_cuda, use_kokkos)
                            test_01(use_cuda, use_kokkos)
                            test_02(use_cuda, use_kokkos)
                            test_03(use_cuda, use_kokkos)
                            test_04(use_cuda, use_kokkos)
                            test_05(use_cuda, use_kokkos)
                            test_06(use_cuda, use_kokkos)


################################################################################
if __name__ == "__main__":
    main()
