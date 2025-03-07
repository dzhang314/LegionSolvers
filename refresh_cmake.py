#!/usr/bin/env python3

import os as _os

from build_utilities import (
    underscore_join,
    remove_directory,
    CMakeDefines,
    cmake,
    SCRATCH_DIR,
)

from build_dependencies import KOKKOS_CUDA_CMAKE_PATH, KOKKOS_NOCUDA_CMAKE_PATH

from build_legion import (
    LEGION_BRANCHES,
    BUILD_TYPES,
    cuda_tag,
    kokkos_tag,
    legion_library_path,
)


def legion_solvers_build_path(
    branch_tag: str, use_cuda: bool, use_kokkos: bool, build_tag: str
) -> str:
    return _os.path.join(
        SCRATCH_DIR,
        "LegionSolversBuild",
        underscore_join(
            branch_tag, cuda_tag(use_cuda), kokkos_tag(use_kokkos), build_tag
        ),
    )


def main() -> None:
    for branch_tag, _, _ in LEGION_BRANCHES:
        for build_tag, build_type in BUILD_TYPES:
            for use_cuda in [False, True]:
                for use_kokkos in [False, True]:
                    legion_path: str = legion_library_path(
                        branch_tag, use_cuda, use_kokkos, build_tag
                    )
                    if not _os.path.isdir(legion_path):
                        continue
                    build_path: str = legion_solvers_build_path(
                        branch_tag, use_cuda, use_kokkos, build_tag
                    )
                    remove_directory(build_path)
                    defines: CMakeDefines = {
                        "CMAKE_CXX_STANDARD": 17,
                        "CMAKE_CXX_FLAGS": "-Wall -Wextra -pedantic -Wfatal-errors",
                        "CMAKE_BUILD_TYPE": build_type,
                        "Legion_DIR": _os.path.join(
                            legion_path,
                            "share",
                            "Legion",
                            "cmake",
                        ),
                    }
                    if use_kokkos:
                        if use_cuda:
                            defines["Kokkos_DIR"] = KOKKOS_CUDA_CMAKE_PATH
                        else:
                            defines["Kokkos_DIR"] = KOKKOS_NOCUDA_CMAKE_PATH
                    cmake(
                        build_path,
                        defines,
                        build=False,
                        test=False,
                        install=False,
                        cmake_cmd=("cmake", _os.getcwd()),
                    )


if __name__ == "__main__":
    main()
