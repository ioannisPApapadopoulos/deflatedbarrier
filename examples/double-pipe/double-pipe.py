# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
from petsc4py import PETSc

"""
Implementation of the Borrvall-Petersson objective functional with a
Stokes PDE constraint. See figure 11 in DOI: 10.1002/fld.426

In this example, we consider the double-pipe which has:
(1) A local minimum (straight channels)
(2) A global minimum (double ended jaw wrench)

"""

width = 1.5 # aspect ratio
N = 50  # mesh resolution # 50, 80, 100

class InflowOutflow(UserExpression):
    """ This class represents the velocity boundary conditions
        as described in Borrvall and Petersson (2003)."""
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0

        if x[0] == 0.0 or x[0] == width:
            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                t = x[1] - 1.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                t = x[1] - 3.0/4
                values[0] = gbar*(1 - (2*t/l)**2)

    def value_shape(self):
        return (2,)

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(width, 1.0), int(width*N),N)
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", mesh.ufl_cell(), 2) # velocity
        Pe = FiniteElement("CG", mesh.ufl_cell(), 1) # pressure
        Ce = FiniteElement("CG", mesh.ufl_cell(), 1) # control
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals

        Ze = MixedElement([Ce, Ve, Pe, Re, Re])
        Z  = FunctionSpace(mesh, Ze)
        self.no_dofs = Z.dim()
        print("Number of degrees of freedom: ", self.no_dofs)

        # Take some data. First, BCs
        self.expr = InflowOutflow(element=Ve)
        # Next, a function space we use to solve for our initial guess
        Ge = MixedElement([Ve, Pe, Re])
        self.G = FunctionSpace(mesh, Ge)
        self.P = FunctionSpace(mesh, Pe)
        return Z

    def expected_inertia(self):
        # dofs of the pressure and equality volume constraint
        return self.P.dim()+1

    def lagrangian(self, z, params):
        rho, u, p, p0, lmbda = split(z)
        (gamma, alpha, q) = params
        L = (
              0.5 * inner(grad(u), grad(u))*dx
            - inner(p, div(u))*dx
            - inner(p0, p)*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            - inner(lmbda, gamma - rho)*dx
            )
        return L

    def number_solutions(self, mu, params):
        return 2

    def cost(self, z, params):
        rho, u, p, p0, lmbda = split(z)
        L = (
              0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), self.expr, "on_boundary")]


    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the Stokes equations for the values of (u, p, p0).
        """
        comm = self.G.mesh().mpi_comm()
        commZ = Z.mesh().mpi_comm()

        def paramstokey(params): return "(" + ", ".join("%.15e" % x for x in params) + ")"
        key = paramstokey(params)
        print("Computing initial guess.")
        gamma = Constant(params[0])
        rho_guess = gamma # Constant(1)

        g = Function(self.G)
        u, p, p0 = split(g)
        v, q, q0 = split(TestFunction(self.G))
        J = self.stokes(u, p, p0, v, q, q0,rho_guess, params)
        F = derivative(J, g, TestFunction(self.G))

        PETScOptions.set("snes_monitor")
        PETScOptions.set("ksp_type", "preonly")
        PETScOptions.set("pc_type", "lu")
        PETScOptions.set("mat_mumps_icntl_14", "1000")
        solver_params = {"nonlinear_solver": "snes"}
        # Compatibility issues
        if PETSc.Sys.getVersion()[0:2] < (3, 9):
            PETScOptions.set("pc_factor_mat_solver_package", "mumps")
        else:
            PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        Gbcs = [DirichletBC(self.G.sub(0), self.expr, "on_boundary")]
        solve(F == 0, g, Gbcs, solver_parameters=solver_params)
        print("Initial guess computed, projecting ...")

        lmbda_guess = Constant(10)
        rho_guess = variable(rho_guess)
        alpha = self.alpha(rho_guess, params)

        ic = as_vector([Constant(gamma), u[0], u[1], p,  p0, lmbda_guess])
        z = project(ic, Z, solver_type="mumps") # there HAS to be a better way to do this

        print("Initial guess projected")
        return [z]

    def solver_parameters(self, mu, branch, task, params):
        (gamma,alphabar, q) = params
        if float(mu) < 100:
            linesearch = "l2"
        else:
            linesearch = "basic"
        if task == 'ContinuationTask':
            max_it = 500
            damping = 1.0
        elif task == 'DeflationTask':
            max_it = 100
            damping = 0.9
        elif task == 'PredictorTask':
            max_it = 20
            damping = 1.0
        args = {
               "snes_max_it": max_it,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_atol": 1e-9,
               "snes_divergence_tolerance": 1.0e10,
               "snes_converged_reason": None,
               "snes_monitor": None,
               "snes_linesearch_type": linesearch,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": damping,
               # "snes_linesearch_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu",
               "pc_factor_mat_solver_type": "mumps",
               "mat_mumps_icntl_24": 1,
               "mat_mumps_icntl_13": 1,
               "mat_mumps_icntl_14": 1000,
               }
        return args

    def alpha(self, rho, params):
        (gamma, alphabar, q) = params
        return alphabar*(1 - rho*(q + 1)/(rho + q))

    def stokes(self, u, p, p0, v, q, q0, rho, params):
        """The Stokes functional, without constraints"""
        J = (
              0.5 * inner(grad(u), grad(u))*dx
            - inner(p, div(u))*dx
            - inner(p0, p)*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        return J

    def bounds_vi(self, Z, mu, params):
        inf = 1e100
        lb = interpolate(Constant((0.0,-inf, -inf, -inf,  -inf, -inf)), Z)
        ub = interpolate(Constant((1.0,+inf, +inf, +inf,  +inf, +inf)), Z)
        return (lb, ub)

    def volume_constraint(self, params):
        return params[0]

    def update_mu(self, u, mu, iters, k, k_mu_old, params):
        k_mu = 0.7
        theta_mu = 1.5
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return feasibletangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)


if __name__ == "__main__":
    problem = BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)

    # Using Benson-Munson solver
    deflatedbarrier(problem, params, mu_start= 100, mu_end = 1e-5, max_halfstep = 0)
