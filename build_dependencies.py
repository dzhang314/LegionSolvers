#!/usr/bin/env python3

from build_utilities import *
import os


GASNET_GIT_URL: str = "https://github.com/StanfordLegion/gasnet.git"
KOKKOS_GIT_URL: str = "https://github.com/kokkos/kokkos.git"


def main():

    with change_directory(LIB_PREFIX):
        remove_directory("gasnet")
        clone(GASNET_GIT_URL)
        with change_directory("gasnet"):
            run("make", "CONDUIT=ibv")  # TODO: machine-to-conduit mapping

    os.chdir(SCRATCH_DIR)

    clone(KOKKOS_GIT_URL, "release-candidate-4.0.0", "kokkos-4.0.0-rc")

    with change_directory("kokkos-4.0.0-rc"):
        defines_cuda: CMakeDefines = {
            "CMAKE_CXX_STANDARD": 17,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-4.0.0-rc-cuda"),
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
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-4.0.0-rc-nocuda"),
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
