#!/usr/bin/env python3

import os
from build_legion_dependencies import remove_directory, cmake, LIB_PREFIX
from build_legion_variants import (
    LEGION_BRANCHES, BUILD_TYPES, NETWORK_TYPES,
    GASNET_DIR, KOKKOS_DIR, join
)


################################################################################


def main():
    for dir_name, branch_name in LEGION_BRANCHES:
        for build_type in BUILD_TYPES:
            for network_tag, _ in NETWORK_TYPES:
                build_name = join("build", dir_name, network_tag, build_type.lower())
                lib_name = join("legion", dir_name, network_tag, build_type.lower())
                remove_directory(build_name)
                cmake(build_name, {
                    "CMAKE_BUILD_TYPE": build_type,
                    "CMAKE_C_COMPILER": "gcc",
                    "CMAKE_CXX_COMPILER": "g++",
                    "GASNet_INCLUDE_DIR": GASNET_DIR,
                    "Kokkos_DIR": KOKKOS_DIR,
                    "Legion_DIR": os.path.join(LIB_PREFIX, lib_name, "share", "Legion", "cmake"),
                }, build=False, test=False, install=False)


################################################################################


if __name__ == "__main__":
    main()
