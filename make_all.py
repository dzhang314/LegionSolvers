#!/usr/bin/env python3

from build_utilities import change_directory, run
from build_legion import LEGION_BRANCHES, BUILD_TYPES
from refresh_cmake import legion_solvers_build_path


def main():
    for branch_tag, _ in LEGION_BRANCHES:
        for build_tag, _ in BUILD_TYPES:
            for use_cuda in [False, True]:
                for use_kokkos in [False, True]:
                    with change_directory(legion_solvers_build_path(
                        branch_tag, use_cuda, use_kokkos, build_tag
                    )):
                        run("cmake", "--build", ".", "--parallel")


if __name__ == "__main__":
    main()
