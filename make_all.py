#!/usr/bin/env python3

import os

from build_utilities import *


################################################################################


DESIRED_VARIANTS = [
    "master_gex_nocuda_nokokkos_debug",
    "master_gex_nocuda_nokokkos_release",
    "master_gex_cuda_nokokkos_debug",
    "master_gex_cuda_nokokkos_release",
    # "master_gex_cuda_kokkos_debug",
    # "master_gex_cuda_kokkos_release",
    "cr_gex_nocuda_nokokkos_debug",
    "cr_gex_nocuda_nokokkos_release",
    "cr_gex_cuda_nokokkos_debug",
    "cr_gex_cuda_nokokkos_release",
    # "cr_gex_cuda_kokkos_debug",
    # "cr_gex_cuda_kokkos_release",
]


def main():
    for branch_tag, branch_name in LEGION_BRANCHES:
        for build_tag, build_type in BUILD_TYPES:
            for network_tag, _ in NETWORK_TYPES:
                for cuda_tag, use_cuda in CUDA_TYPES:
                    for kokkos_tag, use_kokkos in KOKKOS_TYPES:
                        build_name = underscore_join(
                            branch_tag, network_tag,
                            cuda_tag, kokkos_tag, build_tag
                        )
                        if build_name not in DESIRED_VARIANTS:
                            continue
                        build_dir = os.path.join(
                            SCRATCH_DIR,
                            "LegionSolversBuild",
                            build_name
                        )
                        with change_directory(build_dir):
                            run("cmake", "--build", ".", "--parallel", "20")


################################################################################


if __name__ == "__main__":
    main()
