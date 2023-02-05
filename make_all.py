#!/usr/bin/env python3

import os

from build_utilities import *
from build_legion import *


def main():
    for branch_tag, _ in LEGION_BRANCHES:
        for build_tag, _ in BUILD_TYPES:
            for use_cuda in [False, True]:
                for use_kokkos in [False, True]:
                    build_name = underscore_join(
                        branch_tag,
                        cuda_tag(use_cuda), kokkos_tag(use_kokkos), build_tag
                    )
                    build_dir = os.path.join(
                        SCRATCH_DIR,
                        "LegionSolversBuild",
                        build_name
                    )
                    with change_directory(build_dir):
                        run("cmake", "--build", ".", "--parallel")


if __name__ == "__main__":
    main()
