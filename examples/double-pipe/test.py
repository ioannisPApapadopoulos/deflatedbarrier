# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
import sys
doublepipe = __import__("double-pipe")


problem = doublepipe.BorrvallProblem()
params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)

cell_no = 25

class DoublePipe(object):
    def mesh(self, comm):
        mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(1.5, 1.0), int(1.5*cell_no),cell_no)
        return  mesh
    def number_solutions(self, mu, params):
        return 1
    def __getattr__(self, attr):
        return getattr(problem, attr)


newproblem = DoublePipe()
(solutions, out) = deflatedbarrier(newproblem, params, mu_start= 100, mu_end = 1e-5, max_halfstep = 0)
assert len(solutions) == 1

def parameter_update(q, z):
   return q + 0.01
gridsequencing(newproblem,
               sharpness_coefficient = 2,
               branches = [0],
               params = params,
               iters_total = 1,
               parameter_update = parameter_update,
               mu_start_continuation = 1e-5,
               mu_start_refine = 1e-4,
               grid_refinement = 1,
               )
