#!/usr/bin/env python3

import os as _os

from build_utilities import *
from build_dependencies import *


LEGION_GIT_URL: str = "https://gitlab.com/StanfordLegion/legion.git"


def cuda_tag(use_cuda: bool) -> str:
    return "cuda" if use_cuda else "nocuda"


def kokkos_tag(use_kokkos: bool) -> str:
    return "kokkos" if use_kokkos else "nokokkos"


def clone_legion(branch_tag: str, branch_name: str) -> str:
    output_dir = underscore_join("legion", branch_tag)
    remove_directory(output_dir)
    clone(LEGION_GIT_URL, branch=branch_name, path=output_dir)
    return output_dir


def cmake_legion(branch_tag: str, use_cuda: bool, use_kokkos: bool,
                 build_tag: str, build_type: str):
    lib_path = _os.path.join(LIB_PREFIX, underscore_join(
        "legion", branch_tag,
        cuda_tag(use_cuda), kokkos_tag(use_kokkos), build_tag
    ))
    remove_directory(lib_path)
    defines = {
        "CMAKE_CXX_STANDARD": 17,
        "CMAKE_BUILD_TYPE": build_type,
        "CMAKE_INSTALL_PREFIX": lib_path,
        "Legion_BUILD_TESTS": True,
        "Legion_ENABLE_TESTING": True,
        "Legion_USE_OpenMP": True,
        "Legion_USE_CUDA": use_cuda,
        "Legion_NETWORKS": "gasnetex",
        "GASNet_INCLUDE_DIR": _os.path.join(
            LIB_PREFIX, "gasnet", "release", "include"
        ),
    }
    if use_kokkos:
        defines["Legion_USE_Kokkos"] = True
        if use_cuda:
            defines["Kokkos_DIR"] = KOKKOS_CUDA_CMAKE_PATH
            defines["KOKKOS_CXX_COMPILER"] = _os.path.join(
                LIB_PREFIX,
                KOKKOS_CUDA_LIB_NAME if use_cuda else KOKKOS_NOCUDA_LIB_NAME,
                "bin", "nvcc_wrapper"
            )
        else:
            defines["Kokkos_DIR"] = KOKKOS_NOCUDA_CMAKE_PATH
    cmake(underscore_join(
        "build", branch_tag,
        cuda_tag(use_cuda), kokkos_tag(use_kokkos), build_tag
    ), defines, build=True, test=True, install=True)


def main():
    _os.chdir(SCRATCH_DIR)
    for branch_tag, branch_name in LEGION_BRANCHES:
        legion_dir = clone_legion(branch_tag, branch_name)
        with change_directory(legion_dir):
            for build_tag, build_type in BUILD_TYPES:
                for use_cuda, use_kokkos in [(False, False), (False, True),
                                             (True, False), (True, True)]:
                    cmake_legion(branch_tag, use_cuda, use_kokkos,
                                 build_tag, build_type)


if __name__ == "__main__":
    main()
