#!/usr/bin/env python3

import os
import shutil
import subprocess

from build_legion_variants import (
    LEGION_BRANCHES, BUILD_TYPES, NETWORK_TYPES,
    KOKKOS_DIR_FLAG, GASNET_DIR_FLAG, add_tag
)


def main():
    for dir_name, branch_name in LEGION_BRANCHES:
        for build_tag, build_flag in BUILD_TYPES:
            for network_tag, network_flag in NETWORK_TYPES:
                build_name = ["build"]
                lib_name = ["legion"]
                add_tag(build_name, lib_name, dir_name)
                add_tag(build_name, lib_name, network_tag)
                add_tag(build_name, lib_name, build_tag)
                os.chdir("_".join(build_name))
                subprocess.run(["cmake", "--build", ".", "--parallel", "20"])
                os.chdir("..")


if __name__ == "__main__":
    main()
