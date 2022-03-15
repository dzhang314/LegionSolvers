#!/usr/bin/env python3

import os
from build_legion_dependencies import (
    remove_directory, cmake, Machines, MACHINE, LIB_PREFIX
)
from build_legion_variants import (
    LEGION_BRANCHES, BUILD_TYPES, NETWORK_TYPES, KOKKOS_DIR, join
)


################################################################################


def main():
    for dir_name, branch_name in LEGION_BRANCHES:
        for build_type in BUILD_TYPES:
            for network_tag, _ in NETWORK_TYPES:
                build_name = join("build", dir_name, network_tag, build_type.lower())
                lib_name = join("legion", dir_name, network_tag, build_type.lower())
                remove_directory(build_name)
                defines = {
                    "CMAKE_CXX_STANDARD": 17,
                    "CMAKE_BUILD_TYPE": build_type,
                    "Kokkos_DIR": KOKKOS_DIR,
                    "Legion_DIR": os.path.join(LIB_PREFIX, lib_name, "share", "Legion", "cmake"),
                }
                if MACHINE == Machines.LASSEN:
                    defines["CMAKE_CXX_FLAGS"] = "-maltivec -mabi=altivec"
                cmake(build_name, defines, build=False, test=False, install=False)


################################################################################


if __name__ == "__main__":
    main()
