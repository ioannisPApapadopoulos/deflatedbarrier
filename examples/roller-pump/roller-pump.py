# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
from petsc4py import PETSc

"""
Implementation of the Roller-type pump with a
Stokes PDE constraint. See figure 2.19 in DOI: 10.1007/978-981-10-4687-2

Two solutions are found, one avoiding the pump altogether in favor of reaching
the outlet in the shortest distance, whilst the other follows the flow around
the pump.

"""
class InflowOutflow(UserExpression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 0.1
        gbar = 1.0

        if 0.56 < x[0] < 0.66 and x[1] <= 1e-10:
            t = x[0] - 0.61
            values[1] = gbar*(1 - (2*t/l)**2)
        if x[0] >= 1.0-1e-10 and x[1] > 0.9:
            t = x[1] - 0.95
            values[0] = gbar*(1 - (2*t/l)**2)
    def value_shape(self):
        return (2,)

class RollerSpeed(UserExpression):
    def eval(self, values, x):
        values[0] = 10./3.*(x[1] - 0.5)
        values[1] = -10./3.*(x[0] - 0.5)
    def value_shape(self):
        return (2,)

class Dirichlet(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and not ((x[0]-0.5)**2 + (x[1]-0.5)**2 <= 0.31**2))
class DirichletRoller(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and ((x[0]-0.5)**2 + (x[1]-0.5)**2 <= 0.31**2))

class RollerPumpProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        mesh = Mesh("mesh/coarse.xml")

        # Mark boundaries of pump and domain
        sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        sub_domains.set_all(0)
        right = Dirichlet()
        right.mark(sub_domains, 1)
        right = DirichletRoller()
        right.mark(sub_domains, 2)

        print('mesh min size: %s' %mesh.hmin())
        print('mesh max size: %s' %mesh.hmax())
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", triangle, 2) # velocity
        Pe = FiniteElement("CG", triangle, 1) # pressure
        Ce = FiniteElement("CG", triangle, 1) # control
        Re = FiniteElement("R",  triangle, 0) # reals

        Ze = MixedElement([Ce, Ve, Pe, Re, Re])
        Z  = FunctionSpace(mesh, Ze)

        # Take some data. First, BCs
        self.outer = InflowOutflow(element=Ve)
        self.roller = RollerSpeed(element=Ve)
        # Next, a function space we use to solve for our initial guess
        Ge = MixedElement([Ve, Pe, Re])
        self.G = FunctionSpace(mesh, Ge)
        self.Gbcs = [DirichletBC(self.G.sub(0), self.outer, Dirichlet()),
                     DirichletBC(self.G.sub(0), self.roller,DirichletRoller())]
        self.P = FunctionSpace(mesh, Pe)
        self.hmin = mesh.hmin()
        self.no_dofs = Z.dim()
        print("Number of degrees of freedom: ", self.no_dofs)
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

    def cost(self, z, params):
        rho, u, p, p0, lmbda = split(z)
        L = (
              0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), self.outer, Dirichlet()),
                DirichletBC(Z.sub(1), self.roller, DirichletRoller())]


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
        u, p, p0 = split(g)

        J = self.stokes(u, p, p0, rho_guess, params)
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


        solve(F == 0, g, self.Gbcs, solver_parameters=solver_params)
        print("Initial guess computed, projecting ...")

        lmbda_guess = Constant(10)
        rho_guess = variable(rho_guess)
        ic = as_vector([rho_guess, u[0], u[1], p,  p0, lmbda_guess])
        z = project(ic, Z, solver_type="mumps") # there HAS to be a better way to do this

        print("Initial guess projected.")
        return [z]

    def number_solutions(self, mu, params):
        if float(mu) > 7:
            return 1
        else: return 2

    def solver_parameters(self, mu, branch, task, params):
        (gamma,alphabar, q) = params
        linesearch = "l2"
        max_it = 100
        damping = 1.0
        if task == 'DeflationTask':
            linesearch = "basic"
            if self.hmin > 0.02:
                damping = 1.
            else:
                damping = 0.5
            max_it = 300

        args = {
               "snes_max_it": max_it,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_divergence_tolerance": 1.0e30,
               "snes_converged_reason": None,
               "snes_monitor": None,
               "snes_linesearch_type": linesearch,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": damping,
               "snes_linesearch_monitor": None,
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


    def stokes(self, u, p, p0, rho, params):
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
        lb = interpolate(Constant((0.0,-inf, -inf, -inf, -inf, -inf )), Z)
        ub = interpolate(Constant((1.0,+inf, +inf, +inf,  +inf, +inf)), Z)
        return (lb, ub)


    def volume_constraint(self, params):
        return params[0]

    def update_mu(self, u, mu, iters, k, k_mu_old, params):
        k_mu = 0.7
        theta_mu = 2.0
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return feasibletangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)

if __name__ == "__main__":
    problem=RollerPumpProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)

    deflatedbarrier(problem, params, mu_end = 1e-5)

    def parameter_update(q, z):
        return q+0.05
    gridsequencing(problem, sharpness_coefficient = 2, branches = [0],
                   params = params, 
                   parameter_update=parameter_update,
                   iters_total = 11,
                   mu_start_continuation = 1e-5,
                   grid_refinement = 1)
