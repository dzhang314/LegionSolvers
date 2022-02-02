#!/usr/bin/env python3

from contextlib import contextmanager
from enum import Enum
import os
import shutil
import subprocess
import socket


################################################################################


def remove_file(path):
    if os.path.exists(path):
        os.remove(path)


def remove_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)


@contextmanager
def pushd(path):
    prev_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_path)


def run(*command):
    subprocess.run(command, check=True)


def download(url):
    run("wget", url)


def clone(url, branch=None):
    if branch is not None:
        run("git", "clone", "--branch", branch, url)
    else:
        run("git", "clone", url)


def cmake(build_dir="build", defines={}, test=False, install=True):
    os.mkdir(build_dir)
    with pushd(build_dir):
        cmake_cmd = ["cmake", ".."]
        for key, value in defines.items():
            assert isinstance(key, str)
            if isinstance(value, bool):
                cmake_cmd.append("-D{0}={1}".format(key, "ON" if value else "OFF"))
            elif isinstance(value, str) or isinstance(value, int):
                cmake_cmd.append("-D{0}={1}".format(key, value))
            else:
                raise TypeError("Unsupported type: {0}".format(type(value)))
        run(*cmake_cmd)
        run("cmake", "--build", ".", "--parallel", "40")
        if test:
            run("make", "test")
        if install:
            run("make", "install")


############################################################# MACHINE PROPERTIES


class Machines(Enum):
    UNKNOWN = 0
    SAPLING = 1
    LASSEN = 2


HOSTNAME = socket.gethostname()


if HOSTNAME.startswith("lassen"):
    MACHINE = Machines.LASSEN
elif HOSTNAME in ["g0001.stanford.edu", "g0002.stanford.edu",
                  "g0003.stanford.edu", "g0004.stanford.edu"]:
    MACHINE = Machines.SAPLING
else:
    print("WARNING: Running on unknown machine with hostname: " + HOSTNAME)
    MACHINE = Machines.UNKNOWN


if MACHINE == Machines.LASSEN:
    SCRATCH_DIR = "/p/gpfs1/zhang70"
    LIB_PREFIX = "/p/gpfs1/zhang70/lib"
elif MACHINE == Machines.SAPLING:
    SCRATCH_DIR = "/scratch2/dkzhang"
    LIB_PREFIX = "/scratch2/dkzhang/lib"
else:
    SCRATCH_DIR = "/home/dkzhang"
    LIB_PREFIX = "/home/dkzhang/lib"


################################################################################


def main():

    if os.path.exists(LIB_PREFIX):
        print("ERROR: Library directory already exists: " + LIB_PREFIX)
        return
    os.mkdir(LIB_PREFIX)

    with pushd(LIB_PREFIX):
        remove_directory("gasnet")
        clone("https://github.com/StanfordLegion/gasnet.git")
        with pushd("gasnet"):
            run("make", "CONDUIT=ibv")

    os.chdir(SCRATCH_DIR)

    remove_file("3.5.00.zip")
    download("https://github.com/kokkos/kokkos/archive/refs/tags/3.5.00.zip")
    remove_directory("kokkos-3.5.00")
    run("unzip", "3.5.00.zip")
    kokkos_compiler = os.path.join(
        SCRATCH_DIR, "kokkos-3.5.00", "bin", "nvcc_wrapper"
    )
    with pushd("kokkos-3.5.00"):
        cmake("build", {
            "CMAKE_CXX_STANDARD": 17,
            "CMAKE_CXX_EXTENSIONS": True,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_COMPILER": kokkos_compiler,
            "CMAKE_CXX_COMPILER": kokkos_compiler,
            "CMAKE_C_FLAGS": "${CMAKE_C_FLAGS} -DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_CXX_FLAGS": "${CMAKE_CXX_FLAGS} -DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-3.5.00"),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_CUDA": True,
            "Kokkos_ENABLE_CUDA_LAMBDA": True,
            "Kokkos_ENABLE_TESTS": True,
        }, test=True, install=True)

    remove_file("3.0.00.zip")
    download("https://github.com/kokkos/kokkos/archive/refs/tags/3.0.00.zip")
    remove_directory("kokkos-3.0.00")
    run("unzip", "3.0.00.zip")
    kokkos_compiler = os.path.join(
        SCRATCH_DIR, "kokkos-3.0.00", "bin", "nvcc_wrapper"
    )
    with pushd("kokkos-3.0.00"):
        defines = {
            "CMAKE_CXX_STANDARD": 14,
            "CMAKE_CXX_EXTENSIONS": True,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_COMPILER": kokkos_compiler,
            "CMAKE_CXX_COMPILER": kokkos_compiler,
            "CMAKE_C_FLAGS": "${CMAKE_C_FLAGS} -DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_CXX_FLAGS": "${CMAKE_CXX_FLAGS} -DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-3.0.00"),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_CUDA": True,
            "Kokkos_ENABLE_CUDA_LAMBDA": True,
            "Kokkos_ENABLE_TESTS": True,
        }
        # Kokkos 3.0.00 does not auto-detect compute capability
        if MACHINE == Machines.SAPLING:
            defines["Kokkos_ARCH_PASCAL60"] = True
        elif MACHINE == Machines.LASSEN:
            defines["Kokkos_ARCH_VOLTA70"] = True
            defines["Kokkos_ARCH_POWER9"] = True
        cmake("build", defines, test=True, install=True)


################################################################################


if __name__ == "__main__":
    main()
