# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
from petsc4py import PETSc
"""
Five-holes double-pipe example.
Implementation of the Borrvall objective function with a
Navier-Stokes PDE constraint.
"""
class InflowOutflow(UserExpression):
    """ This class represents the velocity boundary conditions
        as described in Borrvall (2003)."""
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0
        if x[0] == 0.0 or x[0] == 1.5:
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
        mesh = Mesh("mesh/mesh.xml")
        return mesh
    def function_space(self, mesh):
        Ve = VectorElement("CG", triangle, 2) # velocity
        Le = VectorElement("CG", triangle, 2) # Lagrange multiplier
        Pe = FiniteElement("CG", triangle, 1) # pressure
        Ce = FiniteElement("CG", triangle, 1) # control
        Re = FiniteElement("R",  triangle, 0) # reals
        Ze = MixedElement([Ce, Ve, Le, Pe, Pe, Re, Re, Re])
        Z  = FunctionSpace(mesh, Ze)
        print("Number of degrees of freedom: ", Z.dim())
        # Take some data. First, BCs
        self.expr = InflowOutflow(element=Ve)
        # Next, a function space we use to solve for our initial guess
        Ge = MixedElement([Ve, Pe, Re])
        self.G = FunctionSpace(mesh, Ge)
        self.Gbcs = [DirichletBC(self.G.sub(0), self.expr, "on_boundary")]
        return Z

    def lagrangian(self, z, params):
        (rho, u, v, p, p_adjoint, p0, p_adjoint_0, lmbda) = split(z)
        (gamma, alphabar, q, delta) = params
        L = (
             # objective function
              0.5 * inner(grad(u), grad(u))*dx
            - inner(p, div(u))*dx
            - inner(p0, p)*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            - inner(lmbda, gamma - rho)*dx
            # adjoint momentum equations
            + inner(grad(u), grad(v))*dx
            + Constant(delta) * inner(dot(u,nabla_grad(u)), v)*dx
            + self.alpha(rho, params) * inner(u, v)*dx
            - inner(p_adjoint, div(v))*dx
            # adjoint pressure nullspace
            - inner(p_adjoint_0, p_adjoint)*dx
            )
        return L

    def cost(self, z, params):
        (rho, u, v, p, q, p0, q0, lmbda) = split(z)
        L = (
              0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), self.expr, "on_boundary"),
                DirichletBC(Z.sub(2), Constant((0.0,0.0)), "on_boundary")]

    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the Stokes equations for the values of (u, p, p0).
        """
        comm = self.G.mesh().mpi_comm()
        commZ = Z.mesh().mpi_comm()
        print("Computing initial guess.")
        gamma = Constant(params[0])
        rho_guess = Constant(1./3.)
        g = Function(self.G)
        (u, p, p0) = split(g)
        (v, q, q0) = split(TestFunction(self.G))
        F = self.navier_stokes(u, p, p0, v, q, q0, rho_guess, params)
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
        solve(F == 0, g, self.Gbcs, solver_parameters = solver_params)
        print("Initial guess computed, projecting ...")
        lmbda_guess = Constant(10)
        rho_guess = variable(rho_guess)
        alpha = self.alpha(rho_guess, params)
        v = Constant((0.0,0.0))
        ic = as_vector([rho_guess, u[0], u[1], v[0], v[1], p,  Constant(0), p0, p0, lmbda_guess])
        z = project(ic, Z, solver_type="mumps")
        print("Initial guess projected.")
        return [z]

    def number_solutions(self, mu, params):
        # To save time in future runs, I know at which values to find branches.
        # For the full experience comment out and just use return 43
        if float(mu) > 150:
            return 1
        elif 120 < float(mu) <= 150:
            return 4
        elif 110 < float(mu) <= 120:
            return 6
        elif 90 < float(mu) <= 110:
            return 9
        elif 80 < float(mu) <= 90:
            return 12
        elif 60 < float(mu) <= 65:
            return 17
        elif 50 < float(mu) <= 60:
            return 19
        elif 43 < float(mu) <= 50:
            return 22
        elif 40 < float(mu) <= 43:
            return 23
        elif 35 < float(mu) <= 40:
            return 24
        elif 30 < float(mu) <= 35:
            return 26
        elif 22 < float(mu) <= 30:
            return 28
        elif 20 < float(mu) <= 22:
            return 34
        elif 19 < float(mu) <= 20:
            return 40
        elif 17 < float(mu) <= 19:
            return 43
        else: return 43

    def solver_parameters(self,mu, branch, task, params):
        (gamma,alphabar, q, density) = params
        linesearch = "l2"
        if task == 'ContinuationTask':
            max_it = 100
            damping = 1.0
        elif task == 'DeflationTask':
            linesearch = "basic"
            max_it = 260
            damping = 0.2
        elif task == 'PredictorTask':
            max_it = 10
            damping = 1.0
        args = {
               "snes_max_it": max_it,
               "snes_atol": 1.0e-9,
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
               "mat_mumps_icntl_13": 1
               }
        return args

    def alpha(self, rho, params):
        (gamma, alphabar, q, delta) = params
        return alphabar*(1 - rho*(q + 1)/(rho + q))

    def navier_stokes(self, u, p, p0, v, q, q0, rho, params):
        """The Navier-Stokes equations"""
        (_,_, _, delta) = params
        F = (
              inner(grad(u), grad(v))*dx
            - inner(p, div(v))*dx
            + Constant(delta) * inner(dot(u,nabla_grad(u)), v)*dx
            + self.alpha(rho, params) * inner(u, v)*dx
            - inner(q, div(u))*dx
            - inner(p0, q)*dx
            - inner(q0, p)*dx
        )

        return F

    def bounds_vi(self, Z, mu, params):
        inf = 1e100
        lb = interpolate(Constant((0.0,-inf, -inf,-inf, -inf, -inf, -inf, -inf, -inf, -inf)), Z)
        ub = interpolate(Constant((1.0,+inf, +inf,+inf, +inf, +inf, +inf, +inf, +inf, +inf)), Z)
        return (lb, ub)

    def volume_constraint(self, params):
        return params[0]

    def update_mu(self, u, mu, iters, k, k_mu_old, params):
        if mu > 130:
            k_mu = 0.8
        else:
            k_mu = 0.9
        theta_mu = 1.1
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return feasibletangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)

if __name__ == "__main__":
    problem=BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1, 1] #(gamma, alphabar, q, delta)

    # Use deflated barrier method to discover solutions
    deflatedbarrier(problem, params, mu_start= 200, max_halfstep = 0)

    def parameter_update(q, z):
        return q + 0.05

    # Grid-sequencing to refine solutions
    for i in range(0,43):
        params = [1.0/3, 2.5e4, 0.1, 1]
        gridsequencing(problem, sharpness_coefficient = 2, branches = [i],
        params = params, iters_total = 3,
        mu_start_refine = 1e-8, grid_refinement = 3,
        parameter_continuation = False)

        # Continuation of sharpness parameter for the poorly refined solutions
        if (i == 25) or (i == 29) or (i == 34) or (i == 37):
            class SharpnessContinuation(object):
                def mesh(self, comm):
                    mesh = Mesh("gs_output/cont-0.1/branch-%s/iter-2/mesh.xml"%i)
                    return  mesh
                def __getattr__(self, attr):
                    return getattr(problem, attr)
            params = [1.0/3, 2.5e4, 0.1, 1]
            newproblem = SharpnessContinuation()

            if (i == 25):
                iters = 2
            elif (i == 37):
                iters = 3
            else: iters = 4

            gridsequencing(newproblem, sharpness_coefficient = 2, branches = [i],
            params = params, iters_total = iters,
            parameter_update=parameter_update, mu_start_refine = 1e-5,
            mu_start_continuation = 1e-5, grid_refinement = 0,
            initialpathfile = "gs_output/cont-0.1/branch-%s/iter-2/%s.xml.gz"%(i,i))
