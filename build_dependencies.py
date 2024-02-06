#!/usr/bin/env python3

import os as _os

from build_utilities import (
    change_directory,
    remove_directory,
    clone,
    download,
    run,
    CMakeDefines,
    cmake,
    LIB_PREFIX,
    SCRATCH_DIR,
    Machines,
    MACHINE,
)


GASNET_GIT_URL: str = "https://github.com/StanfordLegion/gasnet.git"
KOKKOS_GIT_URL: str = "https://github.com/kokkos/kokkos.git"
KOKKOS_4_2_URL: str = "https://github.com/kokkos/kokkos/archive/refs/tags/4.2.00.zip"


_KOKKOS_CUDA_LIB_NAME: str = "kokkos-4.2.00-cuda"
_KOKKOS_NOCUDA_LIB_NAME: str = "kokkos-4.2.00-nocuda"


KOKKOS_CUDA_CMAKE_PATH: str = _os.path.join(
    LIB_PREFIX,
    _KOKKOS_CUDA_LIB_NAME,
    "lib" if MACHINE == Machines.SAPLING else "lib64",
    "cmake",
    "Kokkos",
)

KOKKOS_NOCUDA_CMAKE_PATH: str = _os.path.join(
    LIB_PREFIX,
    _KOKKOS_NOCUDA_LIB_NAME,
    "lib" if MACHINE == Machines.SAPLING else "lib64",
    "cmake",
    "Kokkos",
)


KOKKOS_NVCC_WRAPPER_PATH = _os.path.join(
    LIB_PREFIX, _KOKKOS_CUDA_LIB_NAME, "bin", "nvcc_wrapper"
)


def main():
    with change_directory(LIB_PREFIX):
        remove_directory("gasnet")
        clone(GASNET_GIT_URL, path="gasnet")
        with change_directory("gasnet"):
            run("make", "CONDUIT=ibv")  # TODO: machine-to-conduit mapping

    _os.chdir(SCRATCH_DIR)
    remove_directory("kokkos-4.2.00")
    download(KOKKOS_4_2_URL)
    run("unzip", "4.2.00.zip")

    with change_directory("kokkos-4.2.00"):
        defines_cuda: CMakeDefines = {
            "CMAKE_CXX_STANDARD": 17,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_INSTALL_PREFIX": _os.path.join(LIB_PREFIX, _KOKKOS_CUDA_LIB_NAME),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_CUDA": True,
            "Kokkos_ENABLE_CUDA_LAMBDA": True,
            "Kokkos_ENABLE_CUDA_CONSTEXPR": True,
            "Kokkos_ENABLE_CUDA_LDG_INTRINSIC": True,
            "Kokkos_ENABLE_TESTS": True,
        }
        defines_nocuda: CMakeDefines = {
            "CMAKE_CXX_STANDARD": 17,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_INSTALL_PREFIX": _os.path.join(LIB_PREFIX, _KOKKOS_NOCUDA_LIB_NAME),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_TESTS": True,
        }
        if MACHINE in [Machines.LASSEN, Machines.SUMMIT]:
            defines_nocuda["Kokkos_ARCH_POWER9"] = True
            defines_cuda["Kokkos_ARCH_POWER9"] = True
            defines_cuda["Kokkos_ARCH_VOLTA70"] = True
        cmake("build-cuda", defines_cuda, test=True, install=True)
        cmake("build-nocuda", defines_nocuda, test=True, install=True)


if __name__ == "__main__":
    main()
