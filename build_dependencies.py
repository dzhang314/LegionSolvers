#!/usr/bin/env python3

import os as _os
import sys as _sys

from build_utilities import (
    change_directory,
    remove_directory,
    download,
    run,
    CMakeDefines,
    cmake,
    LIB_PREFIX,
    SCRATCH_DIR,
    Machines,
    MACHINE,
)


KOKKOS_GIT_URL: str = "https://github.com/kokkos/kokkos.git"
KOKKOS_4_5_URL: str = "https://github.com/kokkos/kokkos/archive/refs/tags/4.5.01.zip"


_KOKKOS_CUDA_LIB_NAME: str = "kokkos-4.5.01-cuda"
_KOKKOS_NOCUDA_LIB_NAME: str = "kokkos-4.5.01-nocuda"


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


KOKKOS_NVCC_WRAPPER_PATH: str = _os.path.join(
    LIB_PREFIX, _KOKKOS_CUDA_LIB_NAME, "bin", "nvcc_wrapper"
)


def main() -> None:

    if "--skip-kokkos" not in _sys.argv:

        _os.chdir(SCRATCH_DIR)
        remove_directory("kokkos-4.5.01")
        download(KOKKOS_4_5_URL)
        run("unzip", "4.5.01.zip")

        with change_directory("kokkos-4.5.01"):
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
