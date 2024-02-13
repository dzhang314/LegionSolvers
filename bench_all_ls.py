#!/usr/bin/env python3

import subprocess
from typing import Dict, Iterator, List, Tuple, Union


NUM_CPUS_PER_NODE: int = 40
NUM_GPUS_PER_NODE: int = 4


ProblemSize = Union[int, Tuple[int, int], Tuple[int, int, int]]


def problem_size_iterator_1d() -> Iterator[ProblemSize]:
    nx = 2**12
    while True:
        yield nx
        nx *= 2


def problem_size_iterator_2d() -> Iterator[ProblemSize]:
    nx = 2**6
    ny = 2**6
    while True:
        yield (nx, ny)
        nx *= 2
        yield (nx, ny)
        ny *= 2


def problem_size_iterator_3d() -> Iterator[ProblemSize]:
    nx = 2**4
    ny = 2**4
    nz = 2**4
    while True:
        yield (nx, ny, nz)
        nx *= 2
        yield (nx, ny, nz)
        ny *= 2
        yield (nx, ny, nz)
        nz *= 2


def jsrun_command(
    program: str,
    arguments: List[str],
    rank_per: str = "node",
    use_gpu_aware_mpi: bool = False,
) -> List[str]:
    result = ["jsrun"]
    if use_gpu_aware_mpi:
        result += ["-M", '"-gpu"']
    else:
        result += ["--bind", "none"]
    if rank_per == "node":
        result += ["--rs_per_host", "1"]
        result += ["--cpu_per_rs", str(NUM_CPUS_PER_NODE)]
        result += ["--gpu_per_rs", str(NUM_GPUS_PER_NODE)]
    elif rank_per == "gpu":
        result += ["--rs_per_host", str(NUM_GPUS_PER_NODE)]
        result += ["--cpu_per_rs", str(NUM_CPUS_PER_NODE // NUM_GPUS_PER_NODE)]
        result += ["--gpu_per_rs", "1"]
    else:
        raise ValueError('rank_per must be "node" or "gpu"')
    result += [program]
    result += arguments
    return result


def count_nodes() -> int:
    count_proc = subprocess.run(
        jsrun_command(
            "echo", ["hello"], rank_per="node", use_gpu_aware_mpi=False
        ),
        check=True,
        capture_output=True,
    )
    assert not count_proc.stderr
    count_output = count_proc.stdout.splitlines()
    for line in count_output:
        assert line == b"hello"
    return len(count_output)


NUM_NODES: int = count_nodes()
print("Benchmarking on", NUM_NODES, "nodes.", flush=True)


def petsc_args(
    dim: str,
    problem_size: ProblemSize,
    solver: str,
) -> List[str]:
    result = ["-dim", dim]
    if dim == "1D":
        assert isinstance(problem_size, int)
        result += ["-nx", str(problem_size)]
    elif dim == "2D":
        assert isinstance(problem_size, tuple)
        assert len(problem_size) == 2
        result += ["-nx", str(problem_size[0]), "-ny", str(problem_size[1])]
    elif (dim == "3D") or (dim == "3D27"):
        assert isinstance(problem_size, tuple)
        assert len(problem_size) == 3
        result += [
            "-nx",
            str(problem_size[0]),
            "-ny",
            str(problem_size[1]),
            "-nz",
            str(problem_size[2]),
        ]
    else:
        assert False
    result += ["-ksp_type", solver]
    assert solver in {"cg", "bcgs", "gmres"}
    if solver == "gmres":
        result += ["-ksp_gmres_restart", "10"]
        result += ["-ksp_gmres_haptol", "1.0e-100"]
        result += ["ksp_gmres_modifiedgramschmidt"]
    return result + [
        "-ksp_max_it",
        "200",
        "-pc_type",
        "none",
        "-ksp_atol",
        "1.0e-100",
        "-ksp_rtol",
        "1.0e-100",
        "-ksp_divtol",
        "1.0e+100",
        "-vec_type",
        "cuda",
        "-mat_type",
        "aijcusparse",
    ]


def trilinos_args(
    dim: str,
    problem_size: ProblemSize,
    solver: str,
) -> List[str]:
    assert solver in {"CG", "BiCGStab", "GMRES"}
    if dim == "1D":
        assert isinstance(problem_size, int)
        return [dim, solver, str(problem_size)]
    elif dim == "2D":
        assert isinstance(problem_size, tuple)
        assert len(problem_size) == 2
        return [dim, solver, str(problem_size[0]), str(problem_size[1])]
    elif dim in {"3D", "3D27"}:
        assert isinstance(problem_size, tuple)
        assert len(problem_size) == 3
        return [
            dim,
            solver,
            str(problem_size[0]),
            str(problem_size[1]),
            str(problem_size[2]),
        ]
    else:
        assert False


def legion_solvers_args(
    dim: str,
    problem_size: ProblemSize,
    solver: str,
) -> List[str]:
    result = [
        "-ll:util",
        "4",
        "-ll:gpu",
        str(NUM_GPUS_PER_NODE),
        "-ll:csize",
        "100G",
        "-ll:fsize",
        "12G",
        "-lg:eager_alloc_percentage",
        "20",
        "-dim",
        dim,
        "-solver",
        solver,
    ]
    assert solver in {"1", "2", "3"}
    if dim == "1":
        assert isinstance(problem_size, int)
        result += ["-nx", str(problem_size)]
    elif dim == "2":
        assert isinstance(problem_size, tuple)
        assert len(problem_size) == 2
        result += ["-nx", str(problem_size[0]), "-ny", str(problem_size[1])]
    elif dim in {"3", "4"}:
        assert isinstance(problem_size, tuple)
        assert len(problem_size) == 3
        result += [
            "-nx",
            str(problem_size[0]),
            "-ny",
            str(problem_size[1]),
            "-nz",
            str(problem_size[2]),
        ]
    else:
        assert False
    result += ["-it", "200" if solver != "3" else "25"]
    result += ["-pt", "1"]
    result += ["-vp", str(NUM_GPUS_PER_NODE * NUM_NODES)]
    return result


def benchmark_petsc(
    dim: str, solver: str, problem_size_iterator: Iterator[ProblemSize]
) -> ProblemSize:
    for problem_size in problem_size_iterator:
        num_tries = 0
        while True:
            num_tries += 1
            command = jsrun_command(
                "/g/g20/zhang70/LegionSolvers/benchmarks/petsc/PETScSolverBenchmark",
                petsc_args(dim, problem_size, solver),
                rank_per="gpu",
                use_gpu_aware_mpi=True,
            )
            print(" ".join(command), flush=True)
            proc = subprocess.run(
                command,
                capture_output=True,
            )
            if b"out of memory" in proc.stderr:
                print(
                    ">>> OUT OF MEMORY, FOUND MAX PROBLEM SIZE <<<", flush=True
                )
                return problem_size
            elif proc.returncode == 0:
                print(proc.stdout.decode("utf-8"), flush=True)
                print(">>> SUCCESS, TRYING NEXT PROBLEM SIZE <<<", flush=True)
                break
            elif num_tries >= 5:
                print(proc.stderr.decode("utf-8"), flush=True)
                print(">>> FAILED FIVE TIMES, TERMINATING <<<", flush=True)
                return problem_size
            else:
                print(proc.stderr.decode("utf-8"), flush=True)
                print(">>> ERROR OCCURRED, TRYING AGAIN <<<", flush=True)
    assert False


def benchmark_trilinos(
    dim: str,
    solver: str,
    problem_size_iterator: Iterator[ProblemSize],
    max_problem_size: ProblemSize,
):
    for problem_size in problem_size_iterator:
        if problem_size == max_problem_size:
            return
        num_tries = 0
        while True:
            num_tries += 1
            command = jsrun_command(
                "/g/g20/zhang70/LegionSolvers/benchmarks/trilinos/build/TrilinosSolverBenchmark",
                trilinos_args(dim, problem_size, solver),
                rank_per="gpu",
                use_gpu_aware_mpi=True,
            )
            print(" ".join(command), flush=True)
            proc = subprocess.run(
                command,
                capture_output=True,
            )
            if proc.returncode == 0:
                print(proc.stdout.decode("utf-8"), flush=True)
                print(">>> SUCCESS, TRYING NEXT PROBLEM SIZE <<<", flush=True)
                break
            elif num_tries >= 5:
                print(proc.stderr.decode("utf-8"), flush=True)
                print(">>> FAILED FIVE TIMES, TERMINATING <<<", flush=True)
                return
            else:
                print(proc.stderr.decode("utf-8"), flush=True)
                print(">>> ERROR OCCURRED, TRYING AGAIN <<<", flush=True)


def benchmark_legion_solvers(
    dim: str,
    solver: str,
    problem_size_iterator: Iterator[ProblemSize],
    max_problem_size: ProblemSize,
):
    for problem_size in problem_size_iterator:
        if problem_size == max_problem_size:
            return
        num_tries = 0
        while True:
            num_tries += 1
            command = jsrun_command(
                "/p/gpfs1/zhang70/LegionSolversBuild/cr_cuda_nokokkos_release/BenchmarkStencil",
                legion_solvers_args(dim, problem_size, solver),
                rank_per="node",
                use_gpu_aware_mpi=False,
            )
            print(" ".join(command), flush=True)
            proc = subprocess.run(
                command,
                capture_output=True,
            )
            if proc.returncode == 0:
                print(proc.stdout.decode("utf-8"), flush=True)
                print(">>> SUCCESS, TRYING NEXT PROBLEM SIZE <<<", flush=True)
                break
            elif num_tries >= 5:
                print(proc.stderr.decode("utf-8"), flush=True)
                print(">>> FAILED FIVE TIMES, TERMINATING <<<", flush=True)
                return
            else:
                print(proc.stderr.decode("utf-8"), flush=True)
                print(">>> ERROR OCCURRED, TRYING AGAIN <<<", flush=True)


# MAX_PROBLEM_SIZE_1D_CG = benchmark_petsc("1D", "cg", problem_size_iterator_1d())
# MAX_PROBLEM_SIZE_2D_CG = benchmark_petsc("2D", "cg", problem_size_iterator_2d())
# MAX_PROBLEM_SIZE_3D_CG = benchmark_petsc("3D", "cg", problem_size_iterator_3d())
# MAX_PROBLEM_SIZE_3D27_CG = benchmark_petsc("3D27", "cg", problem_size_iterator_3d())
# MAX_PROBLEM_SIZE_1D_BICGSTAB = benchmark_petsc("1D", "bcgs", problem_size_iterator_1d())
# MAX_PROBLEM_SIZE_2D_BICGSTAB = benchmark_petsc("2D", "bcgs", problem_size_iterator_2d())
# MAX_PROBLEM_SIZE_3D_BICGSTAB = benchmark_petsc("3D", "bcgs", problem_size_iterator_3d())
# MAX_PROBLEM_SIZE_3D27_BICGSTAB = benchmark_petsc(
#     "3D27", "bcgs", problem_size_iterator_3d()
# )
# MAX_PROBLEM_SIZE_1D_GMRES = benchmark_petsc("1D", "gmres", problem_size_iterator_1d())
# MAX_PROBLEM_SIZE_2D_GMRES = benchmark_petsc("2D", "gmres", problem_size_iterator_2d())
# MAX_PROBLEM_SIZE_3D_GMRES = benchmark_petsc("3D", "gmres", problem_size_iterator_3d())
# MAX_PROBLEM_SIZE_3D27_GMRES = benchmark_petsc(
#     "3D27", "gmres", problem_size_iterator_3d()
# )


# benchmark_trilinos("1D", "CG", problem_size_iterator_1d(), MAX_PROBLEM_SIZE_1D_CG)
# benchmark_trilinos("2D", "CG", problem_size_iterator_2d(), MAX_PROBLEM_SIZE_2D_CG)
# benchmark_trilinos("3D", "CG", problem_size_iterator_3d(), MAX_PROBLEM_SIZE_3D_CG)
# benchmark_trilinos("3D27", "CG", problem_size_iterator_3d(), MAX_PROBLEM_SIZE_3D27_CG)
# benchmark_trilinos(
#     "1D", "BiCGStab", problem_size_iterator_1d(), MAX_PROBLEM_SIZE_1D_BICGSTAB
# )
# benchmark_trilinos(
#     "2D", "BiCGStab", problem_size_iterator_2d(), MAX_PROBLEM_SIZE_2D_BICGSTAB
# )
# benchmark_trilinos(
#     "3D", "BiCGStab", problem_size_iterator_3d(), MAX_PROBLEM_SIZE_3D_BICGSTAB
# )
# benchmark_trilinos(
#     "3D27", "BiCGStab", problem_size_iterator_3d(), MAX_PROBLEM_SIZE_3D27_BICGSTAB
# )
# benchmark_trilinos("1D", "GMRES", problem_size_iterator_1d(), MAX_PROBLEM_SIZE_1D_GMRES)
# benchmark_trilinos("2D", "GMRES", problem_size_iterator_2d(), MAX_PROBLEM_SIZE_2D_GMRES)
# benchmark_trilinos("3D", "GMRES", problem_size_iterator_3d(), MAX_PROBLEM_SIZE_3D_GMRES)
# benchmark_trilinos(
#     "3D27", "GMRES", problem_size_iterator_3d(), MAX_PROBLEM_SIZE_3D27_GMRES
# )


MAX_PROBLEM_SIZE_DICT: Dict[int, List[ProblemSize]] = {
    1: [
        1073741824,
        (32768, 32768),
        (1024, 1024, 512),
        (1024, 512, 512),
        1073741824,
        (32768, 16384),
        (1024, 1024, 512),
        (1024, 512, 512),
        536870912,
        (32768, 16384),
        (1024, 1024, 512),
        (1024, 512, 512),
    ],
    2: [
        2147483648,
        (65536, 32768),
        (1024, 1024, 1024),
        (1024, 1024, 512),
        2147483648,
        (32768, 32768),
        (1024, 1024, 1024),
        (1024, 1024, 512),
        1073741824,
        (32768, 32768),
        (1024, 1024, 1024),
        (1024, 1024, 512),
    ],
    4: [
        4294967296,
        (65536, 65536),
        (2048, 1024, 1024),
        (1024, 1024, 1024),
        4294967296,
        (65536, 32768),
        (2048, 1024, 1024),
        (1024, 1024, 1024),
        2147483648,
        (65536, 32768),
        (2048, 1024, 1024),
        (1024, 1024, 1024),
    ],
    8: [
        8589934592,
        (131072, 65536),
        (2048, 2048, 1024),
        (2048, 1024, 1024),
        8589934592,
        (65536, 65536),
        (2048, 2048, 1024),
        (2048, 1024, 1024),
        4294967296,
        (65536, 65536),
        (2048, 2048, 1024),
        (2048, 1024, 1024),
    ],
    16: [
        17179869184,
        (131072, 131072),
        (2048, 2048, 2048),
        (2048, 2048, 1024),
        17179869184,
        (131072, 65536),
        (2048, 2048, 2048),
        (2048, 2048, 1024),
        8589934592,
        (131072, 65536),
        (2048, 2048, 2048),
        (2048, 2048, 1024),
    ],
    32: [
        34359738368,
        (262144, 131072),
        (4096, 2048, 2048),
        (2048, 2048, 2048),
        34359738368,
        (131072, 131072),
        (4096, 2048, 2048),
        (2048, 2048, 2048),
        17179869184,
        (131072, 131072),
        (4096, 2048, 2048),
        (2048, 2048, 2048),
    ],
    64: [
        34359738368,
        (262144, 131072),
        (4096, 4096, 2048),
        (2048, 2048, 2048),
        34359738368,
        (262144, 131072),
        (4096, 4096, 2048),
        (2048, 2048, 2048),
        34359738368,
        (131072, 131072),
        (4096, 2048, 2048),
        (2048, 2048, 2048),
    ],
    128: [
        137438953472,
        (262144, 262144),
        (4096, 4096, 4096),
        (4096, 2048, 2048),
        68719476736,
        (262144, 262144),
        (4096, 4096, 4096),
        (4096, 2048, 2048),
        68719476736,
        (262144, 262144),
        (4096, 4096, 2048),
        (4096, 2048, 2048),
    ],
    256: [
        68719476736,
        (262144, 262144),
        (4096, 4096, 4096),
        (4096, 2048, 2048),
        68719476736,
        (262144, 262144),
        (4096, 4096, 4096),
        (4096, 2048, 2048),
        68719476736,
        (262144, 262144),
        (4096, 4096, 2048),
        (4096, 2048, 2048),
    ],
}


MAX_PROBLEM_SIZES = MAX_PROBLEM_SIZE_DICT[NUM_NODES]


benchmark_legion_solvers(
    "1", "1", problem_size_iterator_1d(), MAX_PROBLEM_SIZES[0]
)
benchmark_legion_solvers(
    "2", "1", problem_size_iterator_2d(), MAX_PROBLEM_SIZES[1]
)
benchmark_legion_solvers(
    "3", "1", problem_size_iterator_3d(), MAX_PROBLEM_SIZES[2]
)
benchmark_legion_solvers(
    "4", "1", problem_size_iterator_3d(), MAX_PROBLEM_SIZES[3]
)
benchmark_legion_solvers(
    "1", "2", problem_size_iterator_1d(), MAX_PROBLEM_SIZES[4]
)
benchmark_legion_solvers(
    "2", "2", problem_size_iterator_2d(), MAX_PROBLEM_SIZES[5]
)
benchmark_legion_solvers(
    "3", "2", problem_size_iterator_3d(), MAX_PROBLEM_SIZES[6]
)
benchmark_legion_solvers(
    "4", "2", problem_size_iterator_3d(), MAX_PROBLEM_SIZES[7]
)
benchmark_legion_solvers(
    "1", "3", problem_size_iterator_1d(), MAX_PROBLEM_SIZES[8]
)
benchmark_legion_solvers(
    "2", "3", problem_size_iterator_2d(), MAX_PROBLEM_SIZES[9]
)
benchmark_legion_solvers(
    "3", "3", problem_size_iterator_3d(), MAX_PROBLEM_SIZES[10]
)
benchmark_legion_solvers(
    "4", "3", problem_size_iterator_3d(), MAX_PROBLEM_SIZES[11]
)


# IGNORED_STDERR_LINES = [
#     "Warning: Overriding spectrum-mpi/rolling-release (module loaded must exactly match)",
#     "Warning: Using      spectrum-mpi/2020.08.19 to match app's MPI",
#     "Please tell John Gyllenhaal (gyllen@llnl.gov, 4-5485) if this MPI env fix doesn't work",
# ]


# print("Benchmarking on", NUM_NODES, "nodes.", flush=True)

# for PROBLEM_SIZE in [2**i for i in range(10, 40)]:

#     finished = False
#     tries = 0
#     while not finished:

#         command = [
#             "jsrun",
#             "--rs_per_host", str(NUM_RANKS_PER_NODE),
#             "--cpu_per_rs", str(NUM_CPU_CORES_PER_NODE // NUM_RANKS_PER_NODE),
#             "--gpu_per_rs", str(NUM_GPUS_PER_RANK),
#             "--bind", "none",
#             "/p/gpfs1/zhang70/LegionSolversBuild/cr_cuda_nokokkos_release/BenchmarkStencil",
#             # "-ll:cpu", str(NUM_CPU_CORES - NUM_UTIL_CORES),
#             "-ll:util", str(NUM_UTIL_CORES_PER_RANK),
#             "-ll:gpu", str(NUM_GPUS_PER_RANK),
#             "-ll:csize", "100G",
#             "-ll:fsize", "12G",
#             "-lg:eager_alloc_percentage", "20",
#             "-nx", str(PROBLEM_SIZE),
#             "-vp", str(NUM_NODES * NUM_RANKS_PER_NODE * NUM_GPUS_PER_RANK),
#             "-it", "1000",
#             # "-lg:prof", str(NUM_NODES * NUM_RANKS_PER_NODE),
#             # "-lg:prof_logfile", f"prof_CSR_CG_{NUM_NODES}_{PROBLEM_SIZE}_%.gz"
#         ]

#         print("Running command:", ' '.join(command), flush=True)
#         proc = subprocess.run(command, capture_output=True)

#         if proc.returncode != 0:
#             print("WARNING: PROCESS RETURNED", proc.returncode)

#         stderr = [
#             line for line in proc.stderr.decode("utf-8").splitlines()
#             if line.strip() and line.strip() not in IGNORED_STDERR_LINES
#         ]

#         if any("default_report_failed_instance_creation" in line for line in stderr):
#             print(PROBLEM_SIZE, " : ", "OUT OF MEMORY!", flush=True)
#             sys.exit(0)

#         if stderr:
#             print(">>>>>>>>>> BEGIN STDERR <<<<<<<<<<", flush=True)
#             for line in stderr:
#                 print(line, flush=True)
#             print(">>>>>>>>>>> END STDERR <<<<<<<<<<<", flush=True)

#         stdout = [
#             line for line in proc.stdout.decode("utf-8").splitlines()
#             if line.strip() and "[LegionSolvers]" not in line
#         ]

#         for line in stdout:
#             if line.startswith("Achieved ") and line.endswith("ms per iteration."):
#                 print(PROBLEM_SIZE, " : ", line[9:-17], flush=True)
#                 finished = True
#                 break
#         else:
#             tries += 1
#             if tries >= 3:
#                 print("ERROR: Tried three times, no success.", flush=True)
#                 finished = True
