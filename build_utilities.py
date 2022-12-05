from contextlib import contextmanager as _contextmanager
from enum import Enum as _Enum
import os as _os
import shutil as _shutil
import subprocess as _subprocess


################################################################################


LEGION_GIT_URL = "https://gitlab.com/StanfordLegion/legion.git"
GASNET_GIT_URL = "https://github.com/StanfordLegion/gasnet.git"


LEGION_BRANCHES = [
    ("master", "master"),
    ("cr", "control_replication"),
]


NETWORK_TYPES = [
    ("", ("Legion_USE_GASNet", True)),
    ("gex", ("Legion_NETWORKS", "gasnetex")),
]


BUILD_TYPES = [("debug", "Debug"), ("release", "RelWithDebInfo")]
CUDA_TYPES = [("cuda", True), ("nocuda", False)]
KOKKOS_TYPES = [("kokkos", True), ("nokokkos", False)]


################################################################################


def quiet_print(quiet, *msg):
    if quiet:
        pass
    else:
        print(*msg)


def underscore_join(*args):
    return '_'.join(arg for arg in args if arg)


def create_directory(path):
    print("[LegionSolversBuild] Creating directory", path)
    assert not _os.path.exists(path)
    _os.mkdir(path)


def remove_file(path, quiet=False):
    if _os.path.exists(path):
        quiet_print(quiet, "[LegionSolversBuild] Removing file", path)
        assert _os.path.isfile(path)
        _os.remove(path)
    else:
        quiet_print(quiet, "[LegionSolversBuild] File", path, "does not exist")


def remove_directory(path, quiet=False):
    if _os.path.exists(path):
        quiet_print(quiet, "[LegionSolversBuild] Removing directory", path)
        assert _os.path.isdir(path)
        _shutil.rmtree(path)
    else:
        quiet_print(quiet, "[LegionSolversBuild] Directory",
                    path, "does not exist")


@_contextmanager
def change_directory(path):
    prev_path = _os.getcwd()
    print("[LegionSolversBuild] Changing directory to", path)
    _os.chdir(path)
    try:
        yield
    finally:
        print("[LegionSolversBuild] Changing directory to", prev_path)
        _os.chdir(prev_path)


def run(*command, check=True):
    print("[LegionSolversBuild] Running command", ' '.join(command))
    _subprocess.run(command, check=check)


################################################################################


def download(url):
    print("[LegionSolversBuild] Downloading file", url)
    run("wget", url)


def clone(url, branch=None):
    if branch is not None:
        print("[LegionSolversBuild] Cloning branch",
              branch, "of repository", url)
        run("git", "clone", "--branch", branch, url)
    else:
        print("[LegionSolversBuild] Cloning repository", url)
        run("git", "clone", url)


def cmake(build_dir="build", defines={}, build=True, test=False, install=True,
          cmake_cmd=("cmake", "..")):
    create_directory(build_dir)
    print("[LegionSolversBuild] Running CMake in directory", build_dir)
    with change_directory(build_dir):
        cmake_cmd = list(cmake_cmd)
        for key, value in defines.items():
            assert isinstance(key, str)
            if isinstance(value, bool):
                cmake_cmd.append(
                    "-D{0}={1}".format(key, "ON" if value else "OFF")
                )
            elif isinstance(value, str) or isinstance(value, int):
                cmake_cmd.append("-D{0}={1}".format(key, value))
            else:
                raise TypeError("Unsupported type: {0}".format(type(value)))
        run(*cmake_cmd)
        if build:
            run("cmake", "--build", ".", "--parallel", "20", check=True)
        if test:
            run("make", "test", check=False)
        if install:
            run("make", "install", check=True)


################################################################################


class Machines(_Enum):
    UNKNOWN = 0
    SAPLING = 1
    PIZDAINT = 2
    LASSEN = 3
    SUMMIT = 4


env_machine = _os.getenv("LEGION_SOLVERS_MACHINE")
assert env_machine is not None

MACHINE = {
    "SAPLING": Machines.SAPLING,
    "PIZDAINT": Machines.PIZDAINT,
    "LASSEN": Machines.LASSEN,
    "SUMMIT": Machines.SUMMIT,
}.get(env_machine.upper(), Machines.UNKNOWN)

if MACHINE == Machines.UNKNOWN:
    print("[LegionSolversBuild] WARNING: Unknown machine")


SCRATCH_DIR = _os.getenv("LEGION_SOLVERS_SCRATCH_DIR")
LIB_PREFIX = _os.getenv("LEGION_SOLVERS_LIB_PREFIX")
assert SCRATCH_DIR is not None
assert LIB_PREFIX is not None
assert _os.path.isdir(SCRATCH_DIR)
assert _os.path.isdir(LIB_PREFIX)


KOKKOS_DIR = {
    True: _os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-cuda",
        "lib64" if MACHINE != Machines.SAPLING else "lib",
        "cmake", "Kokkos"
    ),
    False: _os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-nocuda",
        "lib64" if MACHINE != Machines.SAPLING else "lib",
        "cmake", "Kokkos"
    )
}


KOKKOS_CXX_COMPILER = {
    True: _os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-cuda", "bin", "nvcc_wrapper"
    ),
    False: _os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-nocuda", "bin", "nvcc_wrapper"
    )
}
