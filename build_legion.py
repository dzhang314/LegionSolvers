#!/usr/bin/env python3

import os
import sys

from build_utilities import *


################################################################################


def clone_legion(branch_tag, branch_name):
    output_dir = underscore_join("legion", branch_tag)
    remove_directory("legion")
    remove_directory(output_dir)
    clone(LEGION_GIT_URL, branch=branch_name)
    os.rename("legion", output_dir)
    return output_dir


def cmake_legion(branch_tag, network_tag, network_key, network_val,
                 cuda_tag, use_cuda, kokkos_tag, use_kokkos,
                 build_tag, build_type):
    lib_path = os.path.join(LIB_PREFIX, underscore_join(
        "legion", branch_tag, network_tag,
        cuda_tag, kokkos_tag, build_tag
    ))
    remove_directory(lib_path)
    defines = {
        "CMAKE_CXX_STANDARD": 17,
        "CMAKE_BUILD_TYPE": build_type,
        "CMAKE_INSTALL_PREFIX": lib_path,
        "GASNet_INCLUDE_DIR": os.path.join(LIB_PREFIX, "gasnet", "release", "include"),
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
    cmake(underscore_join(
        "build", branch_tag, network_tag,
        cuda_tag, kokkos_tag, build_tag
    ), defines)


def main():
    os.chdir(SCRATCH_DIR)
    if any(branch_tag in sys.argv for branch_tag, branch_name in LEGION_BRANCHES):
        for branch_tag, branch_name in LEGION_BRANCHES:
            if branch_tag in sys.argv:
                legion_dir = clone_legion(branch_tag, branch_name)
                with change_directory(legion_dir):
                    for build_tag, build_type in BUILD_TYPES:
                        for network_tag, (network_key, network_val) in NETWORK_TYPES:
                            for cuda_tag, use_cuda in CUDA_TYPES:
                                for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                                    cmake_legion(
                                        branch_tag, network_tag,
                                        network_key, network_val,
                                        cuda_tag, use_cuda,
                                        kokkos_tag, use_kokkos,
                                        build_tag, build_type
                                    )
    else:
        for branch_tag, branch_name in LEGION_BRANCHES:
            legion_dir = clone_legion(branch_tag, branch_name)
            with change_directory(legion_dir):
                for build_tag, build_type in BUILD_TYPES:
                    for network_tag, (network_key, network_val) in NETWORK_TYPES:
                        for cuda_tag, use_cuda in CUDA_TYPES:
                            for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                                cmake_legion(
                                    branch_tag, network_tag,
                                    network_key, network_val,
                                    cuda_tag, use_cuda,
                                    kokkos_tag, use_kokkos,
                                    build_tag, build_type
                                )


################################################################################


if __name__ == "__main__":
    main()
