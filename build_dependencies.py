#!/usr/bin/env python3

from build_utilities import *
import os


def main():

    assert MACHINE != Machines.PIZDAINT  # rebuilding dependencies on Piz Daint
    # requires manual modification of some CMake files; don't do it here

    with change_directory(LIB_PREFIX):
        remove_directory("gasnet")
        clone(GASNET_GIT_URL)
        with change_directory("gasnet"):
            run("make", "CONDUIT=ibv")

    os.chdir(SCRATCH_DIR)

    remove_file("3.7.00.zip")
    download("https://github.com/kokkos/kokkos/archive/refs/tags/3.7.00.zip")
    remove_directory("kokkos-3.7.00")
    run("unzip", "3.7.00.zip")
    kokkos_compiler = os.path.join(
        SCRATCH_DIR, "kokkos-3.7.00", "bin", "nvcc_wrapper"
    )
    with change_directory("kokkos-3.7.00"):
        cmake("build-cuda", {
            "CMAKE_CXX_STANDARD": 17,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_COMPILER": kokkos_compiler,
            "CMAKE_CXX_COMPILER": kokkos_compiler,
            "CMAKE_C_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_CXX_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-3.7.00-cuda"),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_CUDA": True,
            "Kokkos_ENABLE_CUDA_LAMBDA": True,
            "Kokkos_ENABLE_TESTS": True,
        }, test=True, install=True)
        cmake("build-nocuda", {
            "CMAKE_CXX_STANDARD": 17,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_CXX_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-3.7.00-nocuda"),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_TESTS": True,
        }, test=True, install=True)
        # tests are known to fail on Sapling and Piz Daint

    remove_file("3.0.00.zip")
    download("https://github.com/kokkos/kokkos/archive/refs/tags/3.0.00.zip")
    remove_directory("kokkos-3.0.00")
    run("unzip", "3.0.00.zip")
    kokkos_compiler = os.path.join(
        SCRATCH_DIR, "kokkos-3.0.00", "bin", "nvcc_wrapper"
    )
    with change_directory("kokkos-3.0.00"):
        defines = {
            "CMAKE_CXX_STANDARD": 14,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_COMPILER": kokkos_compiler,
            "CMAKE_CXX_COMPILER": kokkos_compiler,
            "CMAKE_C_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_CXX_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-3.0.00-cuda"),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_CUDA": True,
            "Kokkos_ENABLE_CUDA_LAMBDA": True,
            "Kokkos_ENABLE_TESTS": True,
        }
        # Kokkos 3.0.00 does not auto-detect compute capability
        if MACHINE in [Machines.SAPLING, Machines.PIZDAINT]:
            defines["Kokkos_ARCH_PASCAL60"] = True
        elif MACHINE in [Machines.LASSEN, Machines.SUMMIT]:
            defines["Kokkos_ARCH_VOLTA70"] = True
            defines["Kokkos_ARCH_POWER9"] = True
        cmake("build-cuda", defines, test=True, install=True)

        defines = {
            "CMAKE_CXX_STANDARD": 14,
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_CXX_FLAGS": "-DKOKKOS_IMPL_TURN_OFF_CUDA_HOST_INIT_CHECK",
            "CMAKE_INSTALL_PREFIX": os.path.join(LIB_PREFIX, "kokkos-3.0.00-nocuda"),
            "Kokkos_ENABLE_SERIAL": True,
            "Kokkos_ENABLE_OPENMP": True,
            "Kokkos_ENABLE_TESTS": True,
        }
        # Kokkos 3.0.00 does not auto-detect compute capability
        if MACHINE in [Machines.LASSEN, Machines.SUMMIT]:
            defines["Kokkos_ARCH_POWER9"] = True
        cmake("build-nocuda", defines, test=True, install=True)
        # tests are known to fail on Piz Daint


################################################################################


if __name__ == "__main__":
    main()
