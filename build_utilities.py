from contextlib import contextmanager as _contextmanager
from enum import Enum as _Enum
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Union as _Union
import os as _os
import shutil as _shutil
import subprocess as _subprocess


def underscore_join(*args: _Any) -> str:
    return "_".join(str(arg) for arg in args if arg)


def _quiet_print(quiet: bool, *msg: _Any) -> None:
    if quiet:
        pass
    else:
        print(*msg)


def create_directory(path: str, quiet: bool = False) -> None:
    _quiet_print(quiet, "[LegionSolversBuild] Creating directory", path)
    assert not _os.path.exists(path)
    _os.mkdir(path)


def remove_file(path: str, quiet: bool = False) -> None:
    if _os.path.exists(path):
        _quiet_print(quiet, "[LegionSolversBuild] Removing file", path)
        assert _os.path.isfile(path)
        _os.remove(path)
    else:
        _quiet_print(quiet, "[LegionSolversBuild] File", path, "does not exist")


def remove_directory(path: str, quiet: bool = False) -> None:
    if _os.path.exists(path):
        _quiet_print(quiet, "[LegionSolversBuild] Removing directory", path)
        assert _os.path.isdir(path)
        _shutil.rmtree(path)
    else:
        _quiet_print(quiet, "[LegionSolversBuild] Directory", path, "does not exist")


@_contextmanager
def change_directory(path: str, quiet: bool = False):
    prev_path: str = _os.getcwd()
    _quiet_print(quiet, "[LegionSolversBuild] Changing directory to", path)
    _os.chdir(path)
    try:
        yield
    finally:
        _quiet_print(quiet, "[LegionSolversBuild] Changing directory to", prev_path)
        _os.chdir(prev_path)


def run(*command: str, check: bool = True, quiet: bool = False) -> None:
    _quiet_print(quiet, "[LegionSolversBuild] Running command", " ".join(command))
    _subprocess.run(command, check=check)


def download(url: str, quiet: bool = False) -> None:
    _quiet_print(quiet, "[LegionSolversBuild] Downloading file", url)
    run("wget", url)


def clone(
    url: str,
    branch: str = "",
    path: str = "",
    commit: str = "",
    quiet: bool = False,
) -> None:
    if branch:
        if path:
            _quiet_print(
                quiet,
                "[LegionSolversBuild] Cloning branch",
                branch,
                "of repository",
                url,
                "into directory",
                path,
            )
            run("git", "clone", "--branch", branch, url, path)
            if commit:
                with change_directory(path):
                    _quiet_print(
                        quiet,
                        "[LegionSolversBuild] Checking out commit",
                        commit,
                        "in directory",
                        path,
                    )
                    run("git", "checkout", commit)
        else:
            _quiet_print(
                quiet,
                "[LegionSolversBuild] Cloning branch",
                branch,
                "of repository",
                url,
            )
            run("git", "clone", "--branch", branch, url)
            assert not commit
    else:
        if path:
            _quiet_print(
                quiet,
                "[LegionSolversBuild] Cloning repository",
                url,
                "into directory",
                path,
            )
            run("git", "clone", url, path)
            if commit:
                with change_directory(path):
                    _quiet_print(
                        quiet,
                        "[LegionSolversBuild] Checking out commit",
                        commit,
                        "in directory",
                        path,
                    )
                    run("git", "checkout", commit)
        else:
            _quiet_print(quiet, "[LegionSolversBuild] Cloning repository", url)
            run("git", "clone", url)
            assert not commit


CMakeDefines = _Dict[str, _Union[bool, int, str]]


def cmake(
    build_path: str = "build",
    defines: CMakeDefines = {},
    build: bool = True,
    test: bool = False,
    install: bool = True,
    cmake_cmd: _Tuple[str, ...] = ("cmake", ".."),
) -> None:
    create_directory(build_path)
    print("[LegionSolversBuild] Running CMake in directory", build_path)
    with change_directory(build_path):
        cmd: _List[str] = list(cmake_cmd)
        for key, value in defines.items():
            if isinstance(value, bool):
                cmd.append("-D{0}={1}".format(key, "ON" if value else "OFF"))
            else:
                cmd.append("-D{0}={1}".format(key, value))
        run(*cmd)
        if build:
            # CMake's automatic CPU core count detection is unreliable on
            # some HPC clusters, launching an unreasonably large number of
            # parallel build jobs. We use Python's os.cpu_count() instead.
            cores = _os.cpu_count()
            if cores is None:
                print("[LegionSolversBuild] WARNING:",
                      "Could not automatically determine CPU core count.")
                cores = 8 # reasonably conservative guess
            print("[LegionSolversBuild] Building with", cores, "cores.")
            run("cmake", "--build", ".", "--parallel", str(cores), check=True)
        if test:
            run("ctest", check=False)
        if install:
            run("make", "install", check=True)


################################################################################


class Machines(_Enum):
    UNKNOWN = 0
    SAPLING = 1
    PIZDAINT = 2
    LASSEN = 3
    SUMMIT = 4


def _getenv(name: str) -> str:
    result = _os.getenv(name)
    if result is None:
        raise RuntimeError("Environment variable " + name + " does not exist")
    return result


MACHINE: Machines = {
    "SAPLING": Machines.SAPLING,
    "PIZDAINT": Machines.PIZDAINT,
    "LASSEN": Machines.LASSEN,
    "SUMMIT": Machines.SUMMIT,
}.get(_getenv("LEGION_SOLVERS_MACHINE").upper(), Machines.UNKNOWN)


if MACHINE == Machines.UNKNOWN:
    print("[LegionSolversBuild] WARNING: Unknown machine")


GASNET_CONDUITS: _Dict[Machines, str] = {
    Machines.UNKNOWN: "mpi",
    Machines.SAPLING: "ibv",
    Machines.PIZDAINT: "mpi", # TODO: don't remember which network Piz Daint uses
    Machines.LASSEN: "ibv",
    Machines.SUMMIT: "ibv",
}


SCRATCH_DIR: str = _getenv("LEGION_SOLVERS_SCRATCH_DIR")
LIB_PREFIX: str = _getenv("LEGION_SOLVERS_LIB_PREFIX")
assert _os.path.isdir(SCRATCH_DIR)
assert _os.path.isdir(LIB_PREFIX)
