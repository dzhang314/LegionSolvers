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


################################################################################


KOKKOS_DIR = {
    True: os.path.join(
        LIB_PREFIX, "kokkos-3.6.00-cuda",
        "lib64" if MACHINE != Machines.SAPLING else "lib",
        "cmake", "Kokkos"
    ),
    False: os.path.join(
        LIB_PREFIX, "kokkos-3.6.00-nocuda",
        "lib64" if MACHINE != Machines.SAPLING else "lib",
        "cmake", "Kokkos"
    )
}


KOKKOS_CXX_COMPILER = {
    True: os.path.join(
        LIB_PREFIX, "kokkos-3.6.00-cuda", "bin", "nvcc_wrapper"
    ),
    False: os.path.join(
        LIB_PREFIX, "kokkos-3.6.00-nocuda", "bin", "nvcc_wrapper"
    )
}


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
                    for cuda_tag, use_cuda in CUDA_TYPES:
                        lib_name = join("legion", dir_name, network_tag, cuda_tag, build_type.lower())
                        remove_directory(os.path.join(LIB_PREFIX, lib_name))
                        defines = {
                            "CMAKE_CXX_STANDARD": 17,
                            "CMAKE_BUILD_TYPE": build_type,
                            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, lib_name),
                            "Legion_EMBED_GASNet": True,
                            "GASNet_CONDUIT": "ibv",
                            "Kokkos_DIR": KOKKOS_DIR[use_cuda],
                            "Legion_USE_OpenMP": True,
                            "Legion_USE_Kokkos": True,
                            network_key: network_val,
                        }
                        if use_cuda:
                            defines["Legion_USE_CUDA"] = True
                            defines["KOKKOS_CXX_COMPILER"] = KOKKOS_CXX_COMPILER[use_cuda]
                        if MACHINE == Machines.PIZDAINT:
                            defines["CUDA_NVCC_FLAGS"] = "-allow-unsupported-compiler"
                        cmake(join("build", dir_name, network_tag, cuda_tag, build_type.lower()), defines)


################################################################################


if __name__ == "__main__":
    main()
