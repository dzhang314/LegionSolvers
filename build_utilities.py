from contextlib import contextmanager as _contextmanager
from enum import Enum as _Enum
import os as _os
import shutil as _shutil
import socket as _socket
import subprocess as _subprocess


################################################################################


LEGION_GIT_URL = "https://gitlab.com/StanfordLegion/legion.git"


LEGION_BRANCHES = [
    ("master", "master"),
    ("cr", "control_replication"),
    ("coll", "collective"),
]


NETWORK_TYPES = [
    ("", ("Legion_USE_GASNet", True)),
    ("gex", ("Legion_NETWORKS", "gasnetex")),
]


BUILD_TYPES = [("debug", "Debug"), ("release", "RelWithDebInfo")]
CUDA_TYPES = [("cuda", True), ("nocuda", False)]
KOKKOS_TYPES = [("kokkos", True), ("nokokkos", False)]


################################################################################


def underscore_join(*args):
    return '_'.join(arg for arg in args if arg)


def create_directory(path):
    print("[LegionSolversBuild] Creating directory", path)
    assert not _os.path.exists(path)
    _os.mkdir(path)


def remove_file(path):
    if _os.path.exists(path):
        print("[LegionSolversBuild] Removing file", path)
        assert _os.path.isfile(path)
        _os.remove(path)
    else:
        print("[LegionSolversBuild] File", path, "does not exist")


def remove_directory(path):
    if _os.path.exists(path):
        print("[LegionSolversBuild] Removing directory", path)
        assert _os.path.isdir(path)
        _shutil.rmtree(path)
    else:
        print("[LegionSolversBuild] Directory", path, "does not exist")


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


HOSTNAME = _socket.gethostname()


if HOSTNAME in ["g0001.stanford.edu", "g0002.stanford.edu",
                "g0003.stanford.edu", "g0004.stanford.edu"]:
    MACHINE = Machines.SAPLING
elif HOSTNAME.startswith("daint") or HOSTNAME.startswith("nid"):
    MACHINE = Machines.PIZDAINT
elif HOSTNAME.startswith("lassen"):
    MACHINE = Machines.LASSEN
elif HOSTNAME.startswith("summit"):
    MACHINE = Machines.SUMMIT
else:
    print("[LegionSolversBuild] WARNING: Unknown machine", HOSTNAME)
    MACHINE = Machines.UNKNOWN


if MACHINE == Machines.LASSEN:
    SCRATCH_DIR = "/p/gpfs1/zhang70"
    LIB_PREFIX = "/p/gpfs1/zhang70/lib"
elif MACHINE == Machines.SAPLING:
    SCRATCH_DIR = "/scratch2/dkzhang"
    LIB_PREFIX = "/scratch2/dkzhang/lib"
elif MACHINE == Machines.PIZDAINT:
    SCRATCH_DIR = "/users/dzhang"
    LIB_PREFIX = "/users/dzhang/lib"
else:
    SCRATCH_DIR = "/home/dkzhang"
    LIB_PREFIX = "/home/dkzhang/lib"


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
