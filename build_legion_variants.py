#!/usr/bin/env python3

import os
import shutil
import subprocess


LIB_PREFIX = "/home/dkzhang/lib"


LEGION_BRANCHES = [
    ("master", "master"),
    ("cr", "control_replication"),
]

BUILD_TYPES = [
    ("debug", "-DCMAKE_BUILD_TYPE=Debug"),
    ("release", "-DCMAKE_BUILD_TYPE=Release"),
]

NETWORK_TYPES = [
    ("", "-DLegion_USE_GASNet=ON"),
    ("gex", "-DLegion_NETWORKS=gasnetex"),
]


KOKKOS_DIR_FLAG = "-DKokkos_DIR=" + LIB_PREFIX + "/kokkos-3.5.00/lib/cmake/Kokkos"
KOKKOS_CXX_COMPILER_FLAG = "-DKOKKOS_CXX_COMPILER=" + LIB_PREFIX + "/kokkos-3.5.00/bin/nvcc_wrapper"
GASNET_DIR_FLAG = "-DGASNet_INCLUDE_DIR=" + LIB_PREFIX + "/gasnet/release/include"


def legion_cmake_command(build_flag, network_flag, lib_name):
    return [
        "cmake", "..",
        "-DCMAKE_INSTALL_PREFIX=" + LIB_PREFIX + "/" + lib_name,
        "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++",
        "-DLegion_USE_OpenMP=ON", "-DLegion_USE_CUDA=ON",
        "-DLegion_MAX_DIM=3", "-DLegion_MAX_FIELDS=512",
        "-DLegion_USE_Kokkos=ON", KOKKOS_DIR_FLAG, KOKKOS_CXX_COMPILER_FLAG,
        network_flag, GASNET_DIR_FLAG, build_flag
    ]


def add_tag(build_name, lib_name, tag):
    if tag:
        build_name.append(tag)
        lib_name.append(tag)


def main():
    for dir_name, branch_name in LEGION_BRANCHES:
        if os.path.exists("legion"):
            shutil.rmtree("legion")
        if os.path.exists("legion_" + dir_name):
            shutil.rmtree("legion_" + dir_name)
        subprocess.run([
            "git", "clone", "--branch", branch_name,
            "https://gitlab.com/StanfordLegion/legion.git"
        ])
        os.rename("legion", "legion_" + dir_name)
        os.chdir("legion_" + dir_name)
        for build_tag, build_flag in BUILD_TYPES:
            for network_tag, network_flag in NETWORK_TYPES:
                build_name = ["build"]
                lib_name = ["legion"]
                add_tag(build_name, lib_name, dir_name)
                add_tag(build_name, lib_name, network_tag)
                add_tag(build_name, lib_name, build_tag)
                build_name = "_".join(build_name)
                lib_name = "_".join(lib_name)
                if os.path.exists(LIB_PREFIX + "/" + lib_name):
                    shutil.rmtree(LIB_PREFIX + "/" + lib_name)
                os.mkdir(build_name)
                os.chdir(build_name)
                subprocess.run(legion_cmake_command(
                    build_flag, network_flag, lib_name
                ))
                subprocess.run(["cmake", "--build", ".", "--parallel", "40"])
                subprocess.run(["make", "install"])
                os.chdir("..")
        os.chdir("..")


if __name__ == "__main__":
    main()
