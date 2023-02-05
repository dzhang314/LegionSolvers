#!/usr/bin/env python3

import os

from build_utilities import *
from build_legion import *


def main():
    LEGION_SOLVERS_DIR = os.getcwd()
    for branch_tag, _ in LEGION_BRANCHES:
        for build_tag, build_type in BUILD_TYPES:
            for use_cuda in [False, True]:
                for use_kokkos in [False, True]:
                    build_name = underscore_join(
                        branch_tag,
                        cuda_tag(use_cuda), kokkos_tag(use_kokkos), build_tag
                    )
                    build_dir = os.path.join(
                        SCRATCH_DIR,
                        "LegionSolversBuild",
                        build_name
                    )
                    lib_name = underscore_join(
                        "legion", branch_tag,
                        cuda_tag(use_cuda), kokkos_tag(use_kokkos), build_tag
                    )
                    remove_directory(build_dir)
                    defines: CMakeDefines = {
                        "CMAKE_CXX_STANDARD": 17,
                        "CMAKE_CXX_FLAGS": ("-Wall -Wextra -pedantic" +
                                            " -Wfatal-errors" +
                                            " -Wno-deprecated-declarations"),
                        "CMAKE_BUILD_TYPE": build_type,
                        "Legion_DIR": os.path.join(
                            LIB_PREFIX, lib_name,
                            "share", "Legion", "cmake"
                        ),
                    }
                    if use_kokkos:
                        if use_cuda:
                            defines["Kokkos_DIR"] = KOKKOS_CUDA_CMAKE_PATH
                        else:
                            defines["Kokkos_DIR"] = KOKKOS_NOCUDA_CMAKE_PATH
                    if branch_tag == "cr":
                        flags = defines["CMAKE_CXX_FLAGS"]
                        assert isinstance(flags, str)
                        defines["CMAKE_CXX_FLAGS"] = flags + \
                            " -DLEGION_SOLVERS_USE_CONTROL_REPLICATION"
                    # if MACHINE == Machines.LASSEN:
                    #     defines["CMAKE_CXX_FLAGS"] += " -maltivec -mabi=altivec"
                    cmake(
                        build_dir, defines,
                        build=False, test=False, install=False,
                        cmake_cmd=("cmake", LEGION_SOLVERS_DIR)
                    )


if __name__ == "__main__":
    main()
