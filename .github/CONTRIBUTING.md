# Contributing to `rising`

If you are interested in contributing to `rising`, you can either implement a new feature or fix a bug.

For both types of contributions, the process is roughly the same:

1. Open an issue in [this repo] and discuss
   the issue with us! Maybe we can give you some hints towards
   implementation/fixing.

1. If you're not part of the core development team, we need you to create your own fork of [this repo], implement it there and create a PR to [this repo] afterwards.

1. Create a new branch (in your fork if necessary) for the implementation of your issue.
   Make sure to include basic unittests.

1. After finishing the implementation, send a pull request to the correct branch of [this repo] (probably master branch).

1. Afterwards, have a look at your pull request since we might suggest some
   changes.

If you are not familiar with creating a pull request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

## Development Install

To develop `rising` on your machine, here are some tips:

1. Uninstall all existing installs of `rising`:

```
pip uninstall rising
pip uninstall rising # run this command twice
```

2. Clone a copy of `rising` from source:

```
git clone https://github.com/PhoenixDL/rising.git
cd rising
```

3. Install `rising` in `build develop` mode:

Install it via

```
python setup.py build develop
```

or

```
pip install -e .
```

This mode will symlink the python files from the current local source tree into the
python install.

Hence, if you modify a python file, you do not need to reinstall `rising`
again and again

In case you want to reinstall, make sure that you uninstall `rising` first by running `pip uninstall rising`
and `python setup.py clean`. Then you can install in `build develop` mode again.

## Code Style

- To improve readability and maintainability, [PEP8 Style](https://www.python.org/dev/peps/pep-0008/) should always be followed
  - maximum code line length is 120
  - maximum doc string line length is 80
- All imports inside the package should be absolute
- If you add a feature, you should also add it to the documentation
- Every module must have an `__all__` section
- All functions should be typed
- Keep functions short and give them meaningful names

## Unit testing

Unittests are located under `tests/`. Run the entire test suite with

```
python -m unittest
```

from the `rising` root directory or run individual test files, like `python test/test_dummy.py`, for individual test suites.

### Better local unit tests with unittest

Testing is done with a `unittest` suite

You can run your tests with coverage by installing ´coverage´ and executing

```bash
coverage run -m unittest; coverage report -m;
```

inside the terminal. Pycharm Professional supports `Run with coverage` directly.
Furthermore, the coverage is always computed and uploaded when you execute `git push` and can be seen on github.

## Writing documentation

`rising` uses an adapted version of [google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Opposing to the original google style we opted to not duplicate the typing from the function
signature to the docstrings.
Length of line inside docstrings block must be limited to 80 characters to
fit into Jupyter documentation popups.

[this repo]: https://github.com/PhoenixDL/rising
