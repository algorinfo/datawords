import os
import shlex
import subprocess
import sys

TORCH_CPU_WHEEL = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA11_6_WHEEL = "https://download.pytorch.org/whl/cu116"


def check_python_version():
    """
    Makes sure that the script is run with Python 3.7 or newer.
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        return
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    raise RuntimeError(
        "Unsupported Python version {}. "
        "Supported versions: 3.7 - 3.11".format(version)
    )


def shell(
    command: str, check=True, input=None, cwd=None, env=None
) -> subprocess.CompletedProcess:
    """
    Runs a provided command, streaming its output to the log files.
    :param command: A command to be executed, as a single string.
    :param check: If true, will throw exception on failure (exit code != 0)
    :param input: Input for the executed command.
    :param cwd: Directory in which to execute the command.
    :param env: A set of environment variable for the process to use.
        If None, the current env is inherited.
    :return: CompletedProcess instance - the result of the command execution.
    """
    proc = subprocess.run(
        shlex.split(command),
        check=check,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        input=input,
        cwd=cwd,
        env=env,
    )

    return proc


def install_pytorch(pkg, pkg_version, cuda_version=None):
    cmd = f"pip install {pkg}=={pkg_version}"
    error = False
    if cuda_version == "11.6":
        print(f"==> Installing {pkg} ver: {pkg_version} with support for cuda 11.6")
        cmd = f"{cmd} --extra-index-url {TORCH_CUDA11_6_WHEEL}"
        msg = f"{pkg}=={pkg_version} cuda:{cuda_version} succesfully installed!!"
    elif cuda_version == "11.7":
        print(f"==> Installing {pkg} ver: {pkg_version} with support for cuda 11.7")
        msg = f"{pkg}=={pkg_version} cuda:{cuda_version} succesfully installed!!"

    elif not cuda_version:
        print(f"==> Installing {pkg} ver: {pkg_version} only for CPU")
        cmd = f"{cmd} --extra-index-url {TORCH_CPU_WHEEL}"
        msg = f"{pkg}=={pkg_version} with CPU succesfully installed!!"
    else:
        error = True
        msg = f"Invalid verion {pkg_version} cuda_version={cuda_version}"

    res = shell(cmd, check=False)
    if res.returncode != 0:
        error = True
        msg = res.stderr.decode()
    return msg, error


if __name__ == "__main__":
    install = os.getenv("INSTALL_TRANSFORMERS")
    if install:
        torch = os.getenv("TORCH_VERSION")
        cuda = os.getenv("CUDA_VERSION")
        res, err = install_pytorch("torch", pkg_version=torch, cuda_version=cuda)
        if err:
            print(res)
            sys.exit(-1)
        print(res)
        sys.exit(0)
    else:
        print("transformers lib not required")
