#!/usr/bin/env python3

import os
import shutil
import subprocess


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


def legion_cmake_command(build_flag, network_flag, lib_name):
    return [
        "cmake", "..",
        "-DCMAKE_INSTALL_PREFIX=/home/dkzhang/lib/" + lib_name,
        "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++",
        "-DLegion_USE_OpenMP=ON", "-DLegion_USE_CUDA=ON",
        "-DLegion_MAX_DIM=4", "-DLegion_MAX_FIELDS=1024",
        "-DLegion_USE_Kokkos=ON",
        "-DKokkos_DIR=/home/dkzhang/lib/kokkos/lib/cmake/Kokkos",
        "-DKOKKOS_CXX_COMPILER=/home/dkzhang/lib/kokkos/bin/nvcc_wrapper",
        "-DGASNet_INCLUDE_DIR=/home/dkzhang/gasnet/release/include",
        network_flag, build_flag
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
                os.mkdir("_".join(build_name))
                os.chdir("_".join(build_name))
                subprocess.run(legion_cmake_command(
                    build_flag, network_flag, "_".join(lib_name)
                ))
                subprocess.run(["make"])
                subprocess.run(["make", "install"])
                os.chdir("..")
        os.chdir("..")


if __name__ == "__main__":
    main()
