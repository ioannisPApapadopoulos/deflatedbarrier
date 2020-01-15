# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
from petsc4py import PETSc

"""
Implementation of the Borrvall-Petersson objective functional with a
Stokes PDE constraint. However with natural boundary conditions on the outlet
rather than a forced Dirichlet boundary boundary_conditions

In this example we find 3 solutions

To demonstrate the flexiblity of FEniCS, we implement Scott-Vogelius elements
"""
width = 1.5 # aspect ratio

class InflowOutflow(UserExpression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0

        if x[0] == 0.0:
            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                t = x[1] - 1.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                t = x[1] - 3.0/4
                values[0] = gbar*(1 - (2*t/l)**2)

    def value_shape(self):
        return (2,)

class Dirichlet(SubDomain):
    def inside(self, x, on_boundary):
        l = 1.0/6.0
        # Dirichlet conditions everywhere but on outlets
        return (on_boundary
                and not (x[0] == width and (1.0/4 - l/2) < x[1] < (1.0/4 + l/2))
                and not (x[0] == width and (3.0/4 - l/2) < x[1] < (3.0/4 + l/2))
                )

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        # Need to use barycentrically refined meshes for inf-sup stability
        mesh = Mesh("mesh/coarse.xml")
        return mesh

    def function_space(self, mesh):
        # Scott-Vogelius FEM
        Ve = VectorElement("CG", mesh.ufl_cell(), 2) # velocity
        Pe = FiniteElement("DG", mesh.ufl_cell(), 1) # pressure
        Ce = FiniteElement("CG", mesh.ufl_cell(), 1) # control
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals

        Ze = MixedElement([Ce, Ve, Pe, Re])
        Z  = FunctionSpace(mesh, Ze)


        # Take some data. First, BCs
        self.expr = InflowOutflow(element=Ve)
        # Next, a function space we use to solve for our initial guess
        Ge = MixedElement([Ve, Pe])
        self.G = FunctionSpace(mesh, Ge)
        self.P = FunctionSpace(mesh, Pe)
        self.no_dofs = Z.dim()
        print("Number of degrees of freedom: ", self.no_dofs)
        return Z


    def expected_inertia(self):
        # dofs of the pressure and equality volume constraint
        return self.P.dim()+1

    def symgrad(self, u):
        return 0.5*(grad(u) + grad(u).T)

    def lagrangian(self, z, params):
        rho, u, p, lmbda = split(z)
        (gamma, alpha, q) = params

        L = (
              inner(self.symgrad(u), self.symgrad(u))*dx
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            - inner(lmbda, gamma - rho)*dx
            )

        return L

    def cost(self, z, params):
        rho, u, p, lmbda = split(z)
        L = (
              # evaluate cost of Borrvall-Petersson functional
              0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), self.expr, Dirichlet())]


    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the Stokes equations for the values of (u, p, p0).
        """
        comm = self.G.mesh().mpi_comm()
        commZ = Z.mesh().mpi_comm()


        print("Computing initial guess.")
        gamma = Constant(params[0])
        rho_guess = gamma
        g = Function(self.G)
        u, p = split(g)
        v, q = split(TestFunction(self.G))
        J = self.stokes(u, p, v, q, rho_guess, params)
        F = derivative(J, g, TestFunction(self.G))

        PETScOptions.set("snes_monitor")
        PETScOptions.set("ksp_type", "preonly")
        PETScOptions.set("pc_type", "lu")
        PETScOptions.set("mat_mumps_icntl_14", "1000")
        solver_params = {"nonlinear_solver": "snes"}
        if PETSc.Sys.getVersion()[0:2] < (3, 9):
            PETScOptions.set("pc_factor_mat_solver_package", "mumps")
        else:
            PETScOptions.set("pc_factor_mat_solver_type", "mumps")

        Gbcs = [DirichletBC(self.G.sub(0), self.expr, Dirichlet())]
        solve(F == 0, g, Gbcs, solver_parameters=solver_params)
        print("Initial guess computed, projecting ...")

        lmbda_guess = Constant(10)
        rho_guess = variable(rho_guess)
        alpha = self.alpha(rho_guess, params)

        ic = as_vector([Constant(gamma), u[0], u[1], p,  lmbda_guess])
        z = project(ic, Z, solver_type="mumps") # there HAS to be a better way to do this

        print("Initial guess projected.")
        return [z]

    def number_solutions(self, mu, params):
        if float(mu) > 130:
            return 1
        elif 40 < float(mu) < 130:
            return 2
        else: return 3


    def solver_parameters(self,mu, branch, task, params):
        (gamma,alphabar, q) = params

        if float(mu) < 100:
            linesearch = "l2"
        else:
            linesearch = "basic"
        if task == 'ContinuationTask':
            max_it = 100
            damping = 1.0
        elif task == 'DeflationTask':
            linesearch = "basic"
            max_it = 200
            damping = 0.9
        elif task == 'PredictorTask':
            max_it = 20
            damping = 1.0

        args = {
               "snes_max_it": max_it,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
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
               }
        if 0 < float(mu) < 90:
             args["snes_atol"] = 1.0e-9
        else:
            args["snes_atol"] = 1.0e-9
        return args

    def alpha(self, rho, params):
        (gamma, alphabar, q) = params
        return alphabar*(1 - rho*(q + 1)/(rho + q))

    def stokes(self, u, p, v, q, rho, params):
        """The Stokes functional, without constraints"""
        J = (
              0.5 * inner(self.symgrad(u), self.symgrad(u))*dx
            - inner(p, div(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        return J

    def bounds_vi(self, Z, mu, params):
        inf = 1e100
        lb = interpolate(Constant((0.0,-inf, -inf, -inf, -inf)), Z)
        ub = interpolate(Constant((1.0,+inf, +inf, +inf, +inf)), Z)
        return (lb, ub)


    def volume_constraint(self, params):
        return params[0]

    def update_mu(self, u, mu, iters, k, k_mu_old, params):
        # rules of IPOPT DOI: 10.1007/s10107-004-0559-y
        k_mu = 0.7
        theta_mu = 1.5
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return feasibletangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)


if __name__ == "__main__":
    problem=BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
    deflatedbarrier(problem, params, mu_start= 1000., mu_end = 1e-5, max_halfstep =0)
