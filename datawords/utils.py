import os

def pkg_route() -> str:
    """
    Get the absoute path of the labfunctions package whatever it is installed
    It could be used to reference files inside of the package.
    :return:
    :type str:
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return here
