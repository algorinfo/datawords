import json
import os
from pathlib import Path

from attrs import asdict as attrs_dict


def pkg_route() -> str:
    """
    Get the absoute path of the labfunctions package whatever it is installed
    It could be used to reference files inside of the package.
    :return:
    :type str:
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return here


def get_version() -> str:
    # fp = pkg_route()
    # with open(f"{fp}/__about__.py", "r") as f:
    #     f.read()
    from datawords.__about__ import __version__

    return __version__


def mkdir_p(fp):
    """Make the fullpath
    similar to mkdir -p in unix systems.
    """
    Path(fp).mkdir(parents=True, exist_ok=True)


def asjson(attr_model):
    _d = attrs_dict(attr_model)
    return json.dumps(_d)

def asdict(attr_model):
    return attrs_dict(attr_model)
