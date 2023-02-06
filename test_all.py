#!/usr/bin/env python3

import subprocess
from typing import Any as _Any
from typing import List as _List

from build_utilities import change_directory
from build_legion import LEGION_BRANCHES, BUILD_TYPES
from refresh_cmake import legion_solvers_build_path


IGNORED_STDERR_LINES: _List[bytes] = [
    b"Warning: Overriding spectrum-mpi/rolling-release (module loaded must exactly match)",
    b"Warning: Using      spectrum-mpi/2020.08.19 to match app's MPI",
    b"Please tell John Gyllenhaal (gyllen@llnl.gov, 4-5485) if this MPI env fix doesn't work"
]


EXPECTED_PARTITION_OUTPUT: _List[bytes] = [
    b"[LegionSolvers] Printing index partition domain_partition with 4 pieces.",
    b"[LegionSolvers] Printing index partition matrix_partition with 4 pieces.",
    b"[LegionSolvers] Printing index partition range_partition with 4 pieces.",
    b"domain_partition (0) <0>",
    b"domain_partition (0) <1>",
    b"domain_partition (0) <2>",
    b"domain_partition (0) <3>",
    b"domain_partition (0) <4>",
    b"domain_partition (0) <5>",
    b"domain_partition (1) <10>",
    b"domain_partition (1) <4>",
    b"domain_partition (1) <5>",
    b"domain_partition (1) <6>",
    b"domain_partition (1) <7>",
    b"domain_partition (1) <8>",
    b"domain_partition (1) <9>",
    b"domain_partition (2) <10>",
    b"domain_partition (2) <11>",
    b"domain_partition (2) <12>",
    b"domain_partition (2) <13>",
    b"domain_partition (2) <14>",
    b"domain_partition (2) <15>",
    b"domain_partition (2) <9>",
    b"domain_partition (3) <14>",
    b"domain_partition (3) <15>",
    b"domain_partition (3) <16>",
    b"domain_partition (3) <17>",
    b"domain_partition (3) <18>",
    b"domain_partition (3) <19>",
    b"matrix_partition (0) <0>",
    b"matrix_partition (0) <10>",
    b"matrix_partition (0) <11>",
    b"matrix_partition (0) <12>",
    b"matrix_partition (0) <13>",
    b"matrix_partition (0) <1>",
    b"matrix_partition (0) <2>",
    b"matrix_partition (0) <3>",
    b"matrix_partition (0) <4>",
    b"matrix_partition (0) <5>",
    b"matrix_partition (0) <6>",
    b"matrix_partition (0) <7>",
    b"matrix_partition (0) <8>",
    b"matrix_partition (0) <9>",
    b"matrix_partition (1) <14>",
    b"matrix_partition (1) <15>",
    b"matrix_partition (1) <16>",
    b"matrix_partition (1) <17>",
    b"matrix_partition (1) <18>",
    b"matrix_partition (1) <19>",
    b"matrix_partition (1) <20>",
    b"matrix_partition (1) <21>",
    b"matrix_partition (1) <22>",
    b"matrix_partition (1) <23>",
    b"matrix_partition (1) <24>",
    b"matrix_partition (1) <25>",
    b"matrix_partition (1) <26>",
    b"matrix_partition (1) <27>",
    b"matrix_partition (1) <28>",
    b"matrix_partition (2) <29>",
    b"matrix_partition (2) <30>",
    b"matrix_partition (2) <31>",
    b"matrix_partition (2) <32>",
    b"matrix_partition (2) <33>",
    b"matrix_partition (2) <34>",
    b"matrix_partition (2) <35>",
    b"matrix_partition (2) <36>",
    b"matrix_partition (2) <37>",
    b"matrix_partition (2) <38>",
    b"matrix_partition (2) <39>",
    b"matrix_partition (2) <40>",
    b"matrix_partition (2) <41>",
    b"matrix_partition (2) <42>",
    b"matrix_partition (2) <43>",
    b"matrix_partition (3) <44>",
    b"matrix_partition (3) <45>",
    b"matrix_partition (3) <46>",
    b"matrix_partition (3) <47>",
    b"matrix_partition (3) <48>",
    b"matrix_partition (3) <49>",
    b"matrix_partition (3) <50>",
    b"matrix_partition (3) <51>",
    b"matrix_partition (3) <52>",
    b"matrix_partition (3) <53>",
    b"matrix_partition (3) <54>",
    b"matrix_partition (3) <55>",
    b"matrix_partition (3) <56>",
    b"matrix_partition (3) <57>",
    b"range_partition (0) <0>",
    b"range_partition (0) <1>",
    b"range_partition (0) <2>",
    b"range_partition (0) <3>",
    b"range_partition (0) <4>",
    b"range_partition (1) <5>",
    b"range_partition (1) <6>",
    b"range_partition (1) <7>",
    b"range_partition (1) <8>",
    b"range_partition (1) <9>",
    b"range_partition (2) <10>",
    b"range_partition (2) <11>",
    b"range_partition (2) <12>",
    b"range_partition (2) <13>",
    b"range_partition (2) <14>",
    b"range_partition (3) <15>",
    b"range_partition (3) <16>",
    b"range_partition (3) <17>",
    b"range_partition (3) <18>",
    b"range_partition (3) <19>",
]


EXPECTED_CG_OUTPUT: _List[bytes] = [
    b"100", b"3280", b"3444", b"3612", b"3784", b"3960",
    b"4140", b"4324", b"4512", b"4704", b"4900"
]


def jsrun_command(program: str, flags: _List[str],
                  use_cuda: bool, use_kokkos: bool,
                  test: bool = True) -> _List[str]:
    # TODO: switch on MACHINE for CPU and GPU numbers
    NUM_CPUS_PER_NODE: int = 40
    NUM_GPUS_PER_NODE: int = 4
    rs_per_host: int = NUM_GPUS_PER_NODE if use_cuda and use_kokkos else 1
    command = [
        "jsrun",
        "--rs_per_host", str(rs_per_host),
        "--cpu_per_rs", str(NUM_CPUS_PER_NODE // rs_per_host),
        "--gpu_per_rs", str(NUM_GPUS_PER_NODE // rs_per_host),
        "--bind", "none",
        program
    ]
    if use_cuda:
        command.extend(["-ll:gpu", str(NUM_GPUS_PER_NODE // rs_per_host)])
    if use_kokkos:
        command.extend(["-ll:ocpu", "1"])
        command.extend(["-ll:othr", "2"])
    if test:
        command.append("-lg:warn")
        command.append("-lg:leaks")
        command.append("-lg:partcheck")
    command.extend(flags)
    return command


def assert_empty(x: _Any):
    if x:
        print("ERROR: Expected to be empty")
        print(x)


def assert_equal(x: _Any, y: _Any):
    if x == y:
        return
    print("ERROR: Expected to be equal")
    print(x)
    print(y)


def test_00(use_cuda: bool, use_kokkos: bool):
    proc = subprocess.run(["Test00Build"], check=True, capture_output=True)
    assert_empty(proc.stderr)
    stdout = proc.stdout.splitlines()
    assert (b"CUDA enabled" if use_cuda else b"CUDA disabled") in stdout
    assert (b"Kokkos enabled" if use_kokkos else b"Kokkos disabled") in stdout
    print("TEST 00 PASSED")


def test_01(use_cuda: bool, use_kokkos: bool):
    proc = subprocess.run(jsrun_command(
        "Test01ScalarOperations", [], use_cuda, use_kokkos, test=True
    ), capture_output=True)
    stdout = [
        line for line in proc.stdout.splitlines()
        if not (line.startswith(b"[LegionSolvers] Registering task") or
                line.startswith(b"[LegionSolvers] Constructing") or
                line.startswith(b"[LegionSolvers] Finished constructing") or
                (b"[warning 1047]" in line) or
                (b"warning_code_1047" in line) or
                (line == b"For more information see:") or
                (line == b""))
    ]
    stderr = [
        line for line in proc.stderr.splitlines()
        if not line in IGNORED_STDERR_LINES
    ]
    assert_empty(stdout)
    assert_empty(stderr)
    print("TEST 01 PASSED" if proc.returncode == 0 else "TEST 01 FAILED")


def test_02(use_cuda: bool, use_kokkos: bool):
    proc = subprocess.run(jsrun_command(
        "Test02VectorOperations", [], use_cuda, use_kokkos, test=True
    ), capture_output=True)
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
    assert_equal(stdout, [b"0", b"0", b"0", b"0", b"0", b"0"])
    assert_empty(stderr)
    print("TEST 02 PASSED" if proc.returncode == 0 else "TEST 02 FAILED")


def test_03(use_cuda: bool, use_kokkos: bool):
    proc = subprocess.run(jsrun_command(
        "Test03COOPartitioning", [], use_cuda, use_kokkos, test=True
    ), capture_output=True)
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
    assert_equal(sorted(set(stdout)), EXPECTED_PARTITION_OUTPUT)
    assert_empty(stderr)
    print("TEST 03 PASSED" if proc.returncode == 0 else "TEST 03 FAILED")


def test_04(use_cuda: bool, use_kokkos: bool):
    proc = subprocess.run(jsrun_command(
        "Test04CSRPartitioning", [], use_cuda, use_kokkos, test=True
    ), capture_output=True)
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
    assert_equal(sorted(set(stdout)), EXPECTED_PARTITION_OUTPUT)
    assert_empty(stderr)
    print("TEST 04 PASSED" if proc.returncode == 0 else "TEST 04 FAILED")


def test_05(use_cuda: bool, use_kokkos: bool):
    proc = subprocess.run(jsrun_command(
        "Test05COOSolveCG", [], use_cuda, use_kokkos, test=True
    ), capture_output=True)
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
    assert_equal(sorted(set(stdout)), EXPECTED_CG_OUTPUT)
    assert_empty(stderr)
    print("TEST 05 PASSED" if proc.returncode == 0 else "TEST 05 FAILED")


def test_06(use_cuda: bool, use_kokkos: bool):
    proc = subprocess.run(jsrun_command(
        "Test06CSRSolveCG", [], use_cuda, use_kokkos, test=True
    ), capture_output=True)
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
    assert_equal(sorted(set(stdout)), EXPECTED_CG_OUTPUT)
    assert_empty(stderr)
    print("TEST 06 PASSED" if proc.returncode == 0 else "TEST 06 FAILED")


def main():
    for branch_tag, _ in LEGION_BRANCHES:
        for build_tag, _ in BUILD_TYPES:
            for use_cuda in [False, True]:
                for use_kokkos in [False, True]:
                    with change_directory(legion_solvers_build_path(
                        branch_tag, use_cuda, use_kokkos, build_tag
                    )):
                        test_00(use_cuda, use_kokkos)
                        test_01(use_cuda, use_kokkos)
                        test_02(use_cuda, use_kokkos)
                        test_03(use_cuda, use_kokkos)
                        test_04(use_cuda, use_kokkos)
                        test_05(use_cuda, use_kokkos)
                        test_06(use_cuda, use_kokkos)


if __name__ == "__main__":
    main()
