#!/usr/bin/env python3

import os
import sys

from build_legion_dependencies import (
    Machines, MACHINE, SCRATCH_DIR, LIB_PREFIX,
    remove_directory, pushd, clone, cmake
)


# LEGION PROPERTIES


LEGION_GIT_URL = "https://gitlab.com/StanfordLegion/legion.git"


LEGION_BRANCHES = [
    ("master", "master"),
    ("cr", "control_replication"),
    ("coll", "collective")
]


BUILD_TYPES = [
    "Debug",
    "Release",
]


NETWORK_TYPES = [
    ("", ("Legion_USE_GASNet", True)),
    ("gex", ("Legion_NETWORKS", "gasnetex")),
]


CUDA_TYPES = [
    ("cuda", True),
    ("nocuda", False),
]


KOKKOS_TYPES = [
    ("kokkos", True),
    ("nokokkos", False),
]


################################################################################


KOKKOS_DIR = {
    True: os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-cuda",
        "lib64" if MACHINE != Machines.SAPLING else "lib",
        "cmake", "Kokkos"
    ),
    False: os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-nocuda",
        "lib64" if MACHINE != Machines.SAPLING else "lib",
        "cmake", "Kokkos"
    )
}


KOKKOS_CXX_COMPILER = {
    True: os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-cuda", "bin", "nvcc_wrapper"
    ),
    False: os.path.join(
        LIB_PREFIX, "kokkos-3.7.00-nocuda", "bin", "nvcc_wrapper"
    )
}


################################################################################


def join(*args):
    return "_".join(arg for arg in args if arg)


def clone_legion(dir_name, branch_name):
    output_dir = join("legion", dir_name)
    remove_directory("legion")
    remove_directory(output_dir)
    clone(LEGION_GIT_URL, branch=branch_name)
    os.rename("legion", output_dir)
    return output_dir


def cmake_legion(dir_name, network_tag, network_key, network_val,
                 cuda_tag, use_cuda, kokkos_tag, use_kokkos, build_type):
    lib_path = os.path.join(LIB_PREFIX, join(
        "legion", dir_name, network_tag,
        cuda_tag, kokkos_tag, build_type.lower()
    ))
    remove_directory(lib_path)
    defines = {
        "CMAKE_CXX_STANDARD": 17,
        "CMAKE_BUILD_TYPE": build_type,
        "CMAKE_INSTALL_PREFIX": lib_path,
        "Legion_EMBED_GASNet": True,
        "GASNet_CONDUIT": "ibv",
        "Legion_USE_OpenMP": True,
        "Legion_USE_CUDA": use_cuda,
        network_key: network_val,
    }
    if use_kokkos:
        defines["Legion_USE_Kokkos"] = True
        defines["Kokkos_DIR"] = KOKKOS_DIR[use_cuda]
        defines["KOKKOS_CXX_COMPILER"] = KOKKOS_CXX_COMPILER[use_cuda]
    if MACHINE == Machines.PIZDAINT:
        defines["CUDA_NVCC_FLAGS"] = "-allow-unsupported-compiler"
    cmake(join(
        "build", dir_name, network_tag,
        cuda_tag, kokkos_tag, build_type.lower()
    ), defines)


def main():
    os.chdir(SCRATCH_DIR)
    if any(dir_name in sys.argv for dir_name, branch_name in LEGION_BRANCHES):
        for dir_name, branch_name in LEGION_BRANCHES:
            if dir_name in sys.argv:
                legion_dir = clone_legion(dir_name, branch_name)
                with pushd(legion_dir):
                    for build_type in BUILD_TYPES:
                        for network_tag, (network_key, network_val) in NETWORK_TYPES:
                            for cuda_tag, use_cuda in CUDA_TYPES:
                                for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                                    cmake_legion(
                                        dir_name, network_tag,
                                        network_key, network_val,
                                        cuda_tag, use_cuda,
                                        kokkos_tag, use_kokkos, build_type
                                    )
    else:
        for dir_name, branch_name in LEGION_BRANCHES:
            legion_dir = clone_legion(dir_name, branch_name)
            with pushd(legion_dir):
                for build_type in BUILD_TYPES:
                    for network_tag, (network_key, network_val) in NETWORK_TYPES:
                        for cuda_tag, use_cuda in CUDA_TYPES:
                            for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                                cmake_legion(
                                    dir_name, network_tag,
                                    network_key, network_val,
                                    cuda_tag, use_cuda,
                                    kokkos_tag, use_kokkos, build_type
                                )


################################################################################


if __name__ == "__main__":
    main()
