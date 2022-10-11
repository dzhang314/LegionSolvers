#!/usr/bin/env python3

import os
from build_legion_dependencies import (
    remove_directory, cmake, Machines, MACHINE, LIB_PREFIX
)
from build_legion_variants import (
    LEGION_BRANCHES, BUILD_TYPES, NETWORK_TYPES, CUDA_TYPES, KOKKOS_TYPES,
    KOKKOS_DIR, join
)


################################################################################


def main():
    for dir_name, branch_name in LEGION_BRANCHES:
        for build_type in BUILD_TYPES:
            for network_tag, _ in NETWORK_TYPES:
                for cuda_tag, use_cuda in CUDA_TYPES:
                    for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                        build_name = os.path.join(
                            join("build", kokkos_tag, cuda_tag),
                            join(dir_name, network_tag, build_type.lower())
                        )
                        lib_name = join(
                            "legion", dir_name, network_tag,
                            cuda_tag, kokkos_tag, build_type.lower()
                        )
                        remove_directory(build_name)
                        defines = {
                            "CMAKE_CXX_STANDARD": 17,
                            "CMAKE_BUILD_TYPE": build_type,
                            "Kokkos_DIR": KOKKOS_DIR[use_cuda],
                            "Legion_DIR": os.path.join(LIB_PREFIX, lib_name, "share", "Legion", "cmake"),
                        }
                        if MACHINE == Machines.LASSEN:
                            defines["CMAKE_CXX_FLAGS"] = "-maltivec -mabi=altivec"
                        cmake(build_name, defines, build=False,
                              test=False, install=False, cmake_cmd=("cmake", ".."))


################################################################################


if __name__ == "__main__":
    main()
