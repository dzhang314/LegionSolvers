#!/usr/bin/env python3

import os

from build_utilities import *


################################################################################


# TODO (rohany, dkzhang): This script should be repurposed to a more general
#  installation script that sets the right cmake flags for the target machine etc.


def main():
    for branch_tag, branch_name in LEGION_BRANCHES:
        for build_tag, build_type in BUILD_TYPES:
            for network_tag, _ in NETWORK_TYPES:
                for cuda_tag, use_cuda in CUDA_TYPES:
                    for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                        build_name = os.path.join(
                            "build",
                            underscore_join(
                                branch_tag, network_tag,
                                cuda_tag, kokkos_tag, build_tag
                            )
                        )
                        lib_name = underscore_join(
                            "legion", branch_tag, network_tag,
                            cuda_tag, kokkos_tag, build_tag
                        )
                        remove_directory(build_name)
                        defines = {
                            "CMAKE_CXX_STANDARD": 17,
                            "CMAKE_CXX_FLAGS": ("-Wall -Wextra -pedantic" +
                                                " -Wfatal-errors -Wno-unused-parameter" +
                                                " -Wno-deprecated-declarations"),
                            "CMAKE_BUILD_TYPE": build_type,
                            "Kokkos_DIR": KOKKOS_DIR[use_cuda],
                            "Legion_DIR": os.path.join(LIB_PREFIX, lib_name, "share", "Legion", "cmake"),
                        }
                        if MACHINE == Machines.LASSEN:
                            defines["CMAKE_CXX_FLAGS"] += " -maltivec -mabi=altivec"
                        cmake(
                            build_name, defines,
                            build=False, test=False, install=False,
                            cmake_cmd=("cmake", os.path.join("..", ".."))
                        )


################################################################################


if __name__ == "__main__":
    main()
