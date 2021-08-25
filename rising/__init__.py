from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__author__ = "Justus Schock, Michael Baumgartner"
__author_email__ = "justus.schock@rwth-aachen.de"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2019-2020, %s." % __author__
__homepage__ = "https://github.com/PhoenixDL/rising"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = (
    "rising is a highly performant, PyTorch only framework for "
    "efficient data augmentation with support for volumetric data"
)
__long_docs__ = ""

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __RISING_SETUP__
except NameError:
    __RISING_SETUP__ = False

if __RISING_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(f"Partial import of `{__name__}` during the build process.\n")  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:
    import sys

    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("once")
    from rising.interface import AbstractMixin
