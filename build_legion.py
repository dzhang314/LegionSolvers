#!/usr/bin/env python3

import os as _os
from typing import List as _List
from typing import Tuple as _Tuple

from build_utilities import (
    underscore_join,
    change_directory,
    remove_directory,
    clone,
    CMakeDefines,
    cmake,
    LIB_PREFIX,
    SCRATCH_DIR,
)

from build_dependencies import (
    KOKKOS_CUDA_CMAKE_PATH,
    KOKKOS_NOCUDA_CMAKE_PATH,
    KOKKOS_NVCC_WRAPPER_PATH,
)


LEGION_GIT_URL: str = "https://gitlab.com/StanfordLegion/legion.git"


LEGION_BRANCHES: _List[_Tuple[str, str]] = [
    ("master", "master"),
    ("cr", "control_replication"),
]


BUILD_TYPES: _List[_Tuple[str, str]] = [
    ("debug", "Debug"),
    ("release", "RelWithDebInfo"),
]


def cuda_tag(use_cuda: bool) -> str:
    return "cuda" if use_cuda else "nocuda"


def kokkos_tag(use_kokkos: bool) -> str:
    return "kokkos" if use_kokkos else "nokokkos"


def clone_legion(branch_tag: str, branch_name: str) -> str:
    output_dir = underscore_join("legion", branch_tag)
    remove_directory(output_dir)
    clone(LEGION_GIT_URL, branch=branch_name, path=output_dir)
    return output_dir


def legion_library_path(
    branch_tag: str, use_cuda: bool, use_kokkos: bool, build_tag: str
) -> str:
    return _os.path.join(
        LIB_PREFIX,
        underscore_join(
            "legion",
            branch_tag,
            cuda_tag(use_cuda),
            kokkos_tag(use_kokkos),
            build_tag,
        ),
    )


def cmake_legion(
    branch_tag: str,
    use_cuda: bool,
    use_kokkos: bool,
    build_tag: str,
    build_type: str,
):
    lib_path = legion_library_path(branch_tag, use_cuda, use_kokkos, build_tag)
    remove_directory(lib_path)
    defines: CMakeDefines = {
        "CMAKE_CXX_STANDARD": 17,
        "CMAKE_BUILD_TYPE": build_type,
        "CMAKE_INSTALL_PREFIX": lib_path,
        "Legion_MAX_NUM_NODES": 4096,
        "Legion_MAX_NUM_PROCS": 256,
        "Legion_USE_OpenMP": True,
        "Legion_USE_CUDA": use_cuda,
        "Legion_NETWORKS": "gasnetex",
        "GASNet_INCLUDE_DIR": _os.path.join(LIB_PREFIX, "gasnet", "release", "include"),
        "CMAKE_CXX_FLAGS": "-DREALM_TIMERS_USE_RDTSC=0",
        "CMAKE_CUDA_FLAGS": "-DREALM_TIMERS_USE_RDTSC=0",
    }
    if use_kokkos:
        defines["Legion_USE_Kokkos"] = True
        if use_cuda:
            defines["Kokkos_DIR"] = KOKKOS_CUDA_CMAKE_PATH
            defines["KOKKOS_CXX_COMPILER"] = KOKKOS_NVCC_WRAPPER_PATH
        else:
            defines["Kokkos_DIR"] = KOKKOS_NOCUDA_CMAKE_PATH
    cmake(
        underscore_join(
            "build",
            branch_tag,
            cuda_tag(use_cuda),
            kokkos_tag(use_kokkos),
            build_tag,
        ),
        defines,
        build=True,
        test=False,
        install=True,
    )


def main():
    _os.chdir(SCRATCH_DIR)
    for branch_tag, branch_name in LEGION_BRANCHES:
        legion_dir = clone_legion(branch_tag, branch_name)
        with change_directory(legion_dir):
            for build_tag, build_type in BUILD_TYPES:
                for use_cuda, use_kokkos in [
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                ]:
                    cmake_legion(
                        branch_tag, use_cuda, use_kokkos, build_tag, build_type
                    )


if __name__ == "__main__":
    main()
