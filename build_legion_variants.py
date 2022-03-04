#!/usr/bin/env python3

import os

from build_legion_dependencies import (
    Machines, MACHINE, SCRATCH_DIR, LIB_PREFIX,
    remove_directory, pushd, clone, cmake
)


############################################################## LEGION PROPERTIES


LEGION_GIT_URL = "https://gitlab.com/StanfordLegion/legion.git"


LEGION_BRANCHES = [
    ("master", "master"),
    ("cr", "control_replication"),
]


BUILD_TYPES = ["Debug", "Release"]


NETWORK_TYPES = [
    ("", ("Legion_USE_GASNet", True)),
    ("gex", ("Legion_NETWORKS", "gasnetex")),
]


################################################################################


KOKKOS_DIR = os.path.join(
    LIB_PREFIX, "kokkos-3.5.00",
    "lib64" if MACHINE != Machines.SAPLING else "lib",
    "cmake", "Kokkos"
)


KOKKOS_CXX_COMPILER = os.path.join(
    LIB_PREFIX, "kokkos-3.5.00", "bin", "nvcc_wrapper"
)


GASNET_DIR = os.path.join(
    LIB_PREFIX, "gasnet", "release", "include"
)


################################################################################


def join(*args):
    return "_".join(arg for arg in args if arg)


def main():
    os.chdir(SCRATCH_DIR)
    for dir_name, branch_name in LEGION_BRANCHES:
        remove_directory("legion")
        remove_directory(join("legion", dir_name))
        clone(LEGION_GIT_URL, branch=branch_name)
        os.rename("legion", join("legion", dir_name))
        with pushd(join("legion", dir_name)):
            for build_type in BUILD_TYPES:
                for network_tag, (network_key, network_val) in NETWORK_TYPES:
                    lib_name = join("legion", dir_name, network_tag, build_type.lower())
                    remove_directory(os.path.join(LIB_PREFIX, lib_name))
                    defines = {
                        "CUDA_NVCC_FLAGS": "-allow-unsupported-compiler",
                        "CMAKE_CXX_EXTENSIONS": True,
                        "CMAKE_BUILD_TYPE": build_type,
                        "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, lib_name),
                        "GASNet_INCLUDE_DIR": GASNET_DIR,
                        "Kokkos_DIR": KOKKOS_DIR,
                        "KOKKOS_CXX_COMPILER": KOKKOS_CXX_COMPILER,
                        "Legion_USE_OpenMP": True,
                        "Legion_USE_CUDA": True,
                        # "Legion_USE_GASNet": True,
                        "Legion_USE_Kokkos": True,
                        # "Legion_MAX_DIM": 3,
                        # "Legion_MAX_FIELDS": 1024,
                        network_key: network_val,
                    }
                    cmake(join("build", dir_name, network_tag, build_type.lower()), defines)


################################################################################


if __name__ == "__main__":
    main()
