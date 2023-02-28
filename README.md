# py-swes: A Python Implementation of a Finite-Difference Solver for the Shallow Water Equations over a Sphere (SWES)

This repository contains a Python coding implementation of Lax-Wendroff-like solver for
the Shallow Water Equations over a Sphere (SWES). All stencils arising from the finite-difference
discretization of the underlying Partial Differential Equations (PDEs) are encoded using the 
embedded Domain Specific Library (eDSL) [GT4Py](https://github.com/GridTools/gt4py).

The code is inspired by the MATLAB code by Dr. P. Connolly, available [here](https://personalpages.manchester.ac.uk/staff/paul.connolly/teaching/practicals/shallow_water_equations.html).


## Quick Start

The solver comes in the shape of the installable package `swes`, whose source code is contained
under `src/`. You can install the package (in editable mode) using pip:

```shell
pip install -e .
```

The folder `scripts/` gathers scripts to run the solver, save the solution in NetCDF format, and
visualize the results. The most relevant configuration options are read from a YAML file.
Template YAML files are available in `config/`.


## References

Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber, P. N. (1992). 
A Standard Test Set for Numerical Approximations to the Shallow Water Equations in Spherical Geometry. 
*Journal of Computational Physics, 102*(1), 211-224.
https://doi.org/10.1016/S0021-9991(05)80016-6