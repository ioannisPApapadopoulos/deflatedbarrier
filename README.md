# README #

### Synopsis ###

DeflAted Barrier method (DAB)

The deflated-barrier library implements the deflated barrier method of Papadopoulos, Farrell and Surowiec. The objective is to compute multiple minima of topology optimization problems which are non-convex, PDE & inequality-constrained optimization problems.

### Dependencies ###


The code is written in Python using FEniCS: a finite element solver platform. Instructions for set up can be found at http://fenicsproject.org/download. (version >=  2017.1.0, compiled with PETSc, petsc4py and HDF5 support).


The deflated-barrier library also depends on

petsc4py (https://bitbucket.org/petsc/petsc4py)
h5py (http://www.h5py.org/)
matplotlib (http://matplotlib.org, for rendering bifurcation diagrams)
mshr (https://bitbucket.org/fenics-project/mshr)
slepc4py (https://bitbucket.org/slepc/slepc4py, for computing stabilities)

### Code examples ###

The easiest way to learn how to use it is to examine the examples in examples/. Start with examples/double-pipe.

### Installation ###

### Contributors ###

Ioannis P. A. Papadopoulos (ioannis.papadopoulos@maths.ox.ac.uk)
Patrick E. Farrell (patrick.farrell@maths.ox.ac.uk)


### Disclaimer ###
