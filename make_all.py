#!/usr/bin/env python3

import os
from build_legion_dependencies import run
from build_legion_variants import (
    LEGION_BRANCHES, BUILD_TYPES, NETWORK_TYPES, CUDA_TYPES, KOKKOS_TYPES,
    join, pushd
)


################################################################################


def main():
    for dir_name, branch_name in LEGION_BRANCHES:
        for build_type in BUILD_TYPES:
            for network_tag, _ in NETWORK_TYPES:
                for cuda_tag, use_cuda in CUDA_TYPES:
                    for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                        if use_kokkos and not use_cuda:
                            continue
                        build_name = os.path.join(
                            "build",
                            join(
                                dir_name, network_tag, cuda_tag,
                                kokkos_tag, build_type.lower()
                            )
                        )
                        with pushd(build_name):
                            run("cmake", "--build", ".", "--parallel", "20")

################################################################################


if __name__ == "__main__":
    main()
