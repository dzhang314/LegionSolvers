#!/usr/bin/env python3

import os as _os

from build_utilities import change_directory, run
from build_legion import LEGION_BRANCHES, BUILD_TYPES
from refresh_cmake import legion_solvers_build_path


def main() -> None:
    for branch_tag, _, _ in LEGION_BRANCHES:
        for build_tag, _ in BUILD_TYPES:
            for use_cuda in [False, True]:
                for use_kokkos in [False, True]:
                    build_path: str = legion_solvers_build_path(
                        branch_tag, use_cuda, use_kokkos, build_tag
                    )
                    if not _os.path.isdir(build_path):
                        continue
                    with change_directory(build_path):
                        run("cmake", "--build", ".", "--parallel")


if __name__ == "__main__":
    main()
