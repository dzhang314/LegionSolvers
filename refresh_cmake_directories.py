#!/usr/bin/env python3

import os
import shutil
import subprocess

from build_legion_variants import (
    LEGION_BRANCHES, BUILD_TYPES, NETWORK_TYPES, add_tag
)


def legion_solvers_cmake_command(build_flag, lib_name):
    return [
        "cmake", "..",
        "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++",
        "-DLegion_DIR=/home/dkzhang/lib/" + lib_name + "/share/Legion/cmake",
        "-DGASNet_INCLUDE_DIR=/home/dkzhang/gasnet/release/include",
        "-DKokkos_DIR=/home/dkzhang/lib/kokkos/lib/cmake/Kokkos",
        build_flag
    ]


def main():
    for dir_name, branch_name in LEGION_BRANCHES:
        for build_tag, build_flag in BUILD_TYPES:
            for network_tag, network_flag in NETWORK_TYPES:
                build_name = ["build"]
                lib_name = ["legion"]
                add_tag(build_name, lib_name, dir_name)
                add_tag(build_name, lib_name, network_tag)
                add_tag(build_name, lib_name, build_tag)
                build_name = "_".join(build_name)
                if os.path.exists(build_name):
                    shutil.rmtree(build_name)
    for dir_name, branch_name in LEGION_BRANCHES:
        for build_tag, build_flag in BUILD_TYPES:
            for network_tag, network_flag in NETWORK_TYPES:
                build_name = ["build"]
                lib_name = ["legion"]
                add_tag(build_name, lib_name, dir_name)
                add_tag(build_name, lib_name, network_tag)
                add_tag(build_name, lib_name, build_tag)
                build_name = "_".join(build_name)
                lib_name = "_".join(lib_name)
                if os.path.exists(build_name):
                    shutil.rmtree(build_name)
                os.mkdir(build_name)
                os.chdir(build_name)
                subprocess.run(legion_solvers_cmake_command(
                    build_flag, lib_name
                ))
                os.chdir("..")


if __name__ == "__main__":
    main()
