import os
from pathlib import Path


def pkg_route() -> str:
    """
    Get the absoute path of the labfunctions package whatever it is installed
    It could be used to reference files inside of the package.
    :return:
    :type str:
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return here


def mkdir_p(fp):
    """Make the fullpath
    similar to mkdir -p in unix systems.
    """
    Path(fp).mkdir(parents=True, exist_ok=True)
