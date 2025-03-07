#!/usr/bin/env python3

import os as _os
import sys as _sys
from typing import List as _List
from typing import Tuple as _Tuple

from build_utilities import (
    underscore_join,
    change_directory,
    remove_directory,
    clone,
    CMakeDefines,
    cmake,
    LIB_PREFIX,
    SCRATCH_DIR,
    Machines,
    MACHINE,
    GASNET_CONDUITS,
)

from build_dependencies import (
    KOKKOS_CUDA_CMAKE_PATH,
    KOKKOS_NOCUDA_CMAKE_PATH,
    KOKKOS_NVCC_WRAPPER_PATH,
)


LEGION_GIT_URL: str = "https://gitlab.com/StanfordLegion/legion.git"


LEGION_BRANCHES: _List[_Tuple[str, str, str]] = [
    # ("aug22-3", "master", "ef074af5298f2259f1958918e1d9ba0e14bd7876"), # works
    # ("aug22-4", "master", "4b7981f53b800e409f3753878c37fe75e4ed7449"), # /usr/bin/ld: cannot find -lrealm_gex_wrapper_objs
    # ("sep12", "master", "c032dab254f423ccab36d05c47fed42b94f0b3f5"), # /usr/bin/ld: cannot find -lrealm_gex_wrapper_objs
    ("master", "master", "master"),
    ("r2412", "legion-24.12.0", "legion-24.12.0"),
]


BUILD_TYPES: _List[_Tuple[str, str]] = [
    ("debug", "Debug"),
    ("release", "RelWithDebInfo"),
]


def cuda_tag(use_cuda: bool) -> str:
    return "cuda" if use_cuda else "nocuda"


def kokkos_tag(use_kokkos: bool) -> str:
    return "kokkos" if use_kokkos else "nokokkos"


def clone_legion(branch_tag: str, branch_name: str, commit: str) -> str:
    output_dir: str = underscore_join("legion", branch_tag)
    remove_directory(output_dir)
    clone(LEGION_GIT_URL, branch=branch_name, path=output_dir, commit=commit)
    return output_dir


def legion_library_path(
    branch_tag: str, use_cuda: bool, use_kokkos: bool, build_tag: str
) -> str:
    return _os.path.join(
        LIB_PREFIX,
        underscore_join(
            "legion",
            branch_tag,
            cuda_tag(use_cuda),
            kokkos_tag(use_kokkos),
            build_tag,
        ),
    )


def cmake_legion(
    branch_tag: str,
    use_cuda: bool,
    use_kokkos: bool,
    build_tag: str,
    build_type: str,
) -> None:
    lib_path: str = legion_library_path(branch_tag, use_cuda, use_kokkos, build_tag)
    remove_directory(lib_path)
    defines: CMakeDefines = {
        "CMAKE_CXX_STANDARD": 17,
        "CMAKE_BUILD_TYPE": build_type,
        "CMAKE_INSTALL_PREFIX": lib_path,
        "Legion_MAX_NUM_NODES": 4096,
        "Legion_MAX_NUM_PROCS": 256,
        "Legion_USE_OpenMP": True,
        "Legion_USE_CUDA": use_cuda,
        "Legion_NETWORKS": "gasnetex",
        "Legion_EMBED_GASNet": True,
        "GASNet_CONDUIT": GASNET_CONDUITS[MACHINE],
    }
    if MACHINE in [Machines.LASSEN, Machines.SUMMIT]:
        # Disable rdtsc instruction on non-x86 machines.
        defines["CMAKE_CXX_FLAGS"] = "-DREALM_TIMERS_USE_RDTSC=0"
        defines["CMAKE_CUDA_FLAGS"] = "-DREALM_TIMERS_USE_RDTSC=0"
    if use_kokkos:
        defines["Legion_USE_Kokkos"] = True
        if use_cuda:
            defines["Kokkos_DIR"] = KOKKOS_CUDA_CMAKE_PATH
            defines["KOKKOS_CXX_COMPILER"] = KOKKOS_NVCC_WRAPPER_PATH
        else:
            defines["Kokkos_DIR"] = KOKKOS_NOCUDA_CMAKE_PATH
    cmake(
        underscore_join(
            "build",
            branch_tag,
            cuda_tag(use_cuda),
            kokkos_tag(use_kokkos),
            build_tag,
        ),
        defines,
        build=True,
        test=False,
        install=True,
    )


def main() -> None:
    cuda_kokkos_configs: _List[_Tuple[bool, bool]] = [
        (False, False), (False, True), (True, False), (True, True),
    ]
    if "--force-cuda" in _sys.argv:
        cuda_kokkos_configs = [c for c in cuda_kokkos_configs if c[0]]
    if "--skip-cuda" in _sys.argv:
        cuda_kokkos_configs = [c for c in cuda_kokkos_configs if not c[0]]
    if "--force-kokkos" in _sys.argv:
        cuda_kokkos_configs = [c for c in cuda_kokkos_configs if c[1]]
    if "--skip-kokkos" in _sys.argv:
        cuda_kokkos_configs = [c for c in cuda_kokkos_configs if not c[1]]
    _os.chdir(SCRATCH_DIR)
    for branch_tag, branch_name, commit in LEGION_BRANCHES:
        legion_dir: str = clone_legion(branch_tag, branch_name, commit)
        with change_directory(legion_dir):
            for build_tag, build_type in BUILD_TYPES:
                for use_cuda, use_kokkos in cuda_kokkos_configs:
                    cmake_legion(
                        branch_tag, use_cuda, use_kokkos, build_tag, build_type
                    )


if __name__ == "__main__":
    main()
