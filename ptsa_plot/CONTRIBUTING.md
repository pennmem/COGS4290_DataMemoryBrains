# Contributing to ptsa_plot

## General guidelines

- Always use docstrings for classes and functions.
  [Numpy-style](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
  doc strings are preferred.
- It's nice to include demo usages whether in docstrings or as a standalone
  example file.

## How to contribute new code

1. Fork this repository and create a new branch
2. Add your new code to a sensible place
3. Follow [PEP8][] guidelines as best as possible (it's OK to ignore the line
   length limit of 80 characters)
4. Add unit tests to ensure that everything works as expected. Test files should
   be placed in the `ptsa_plot.test` package and be named `test_yournewmodule.py`
   where `yournewmodule` is the name of your new module.
5. Submit a pull request
6. Make modifications as requested by the maintainers

Your new code should run on Python 3 and legacy Python 2.7. Some exceptions may
exist if dependencies for some reason are not compatible with certain Python
versions (you should seriously consider if it's a necessary dependency if that
is the case).

[PEP8]: https://www.python.org/dev/peps/pep-0008/
