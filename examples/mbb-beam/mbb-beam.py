# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
from petsc4py import PETSc

"""
MBB beam example, 2 minimizers found
"""

class South(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[1] == 0.0)  and (x[0] >=  2.9) and on_boundary)

class West(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] == 0.0) and on_boundary

class North(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] <= 0.1) and (x[1] == 1.0) and on_boundary

class MBBProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(3.0, 1.0), int(3*50),50)
        return mesh

    def boundary_ds(self, mesh):
        North_boundary = North()
        boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary_parts.set_all(0)
        North_boundary.mark(boundary_parts, 1)
        ds = Measure("ds")(subdomain_data=boundary_parts)
        return ds

    def function_space(self, mesh):
        Ve = VectorElement("CG", mesh.ufl_cell(), 1) # velocity
        Ce = FiniteElement("CG", mesh.ufl_cell(), 1) # control
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals

        Ze = MixedElement([Ce, Ve, Re])
        Z  = FunctionSpace(mesh, Ze)

        # Take some data. First, BCs
        # Next, a function space we use to solve for our initial guess

        self.G = FunctionSpace(mesh, Ve)
        self.Gbcs = [DirichletBC(self.G.sub(1), 0.0, South()),
                     DirichletBC(self.G.sub(0), 0.0, West())]

        return Z

    def expected_inertia(self):
        expected = self.G.dim() # velocity multiplier contribution
        expected += 1           # volume constraint multiplier contribution

        # Boundary conditions fix dofs on velocity multiplier and they do
        # not contribute to the negative eigenvalue count.
        u = Function(self.G)
        u.assign(Constant((1.0,1.0)))
        [bc.apply(u.vector()) for bc in self.Gbcs]
        dofs = len(where(u.vector()==0.0)[0])

        expected -= dofs        # velocity multiplier boundary conditions contribution
        return expected

    def symgrad(self, u):
        return 0.5*(grad(u) + grad(u).T)

    def lagrangian(self, z, params):

        ds = self.boundary_ds(z.function_space().mesh())
        rho, u, lmbda = split(z)
        (gamma, p, eps, f, mu_lame, lmbda_lame, epsilon) = params
        f = Constant((0,f))
        I = Identity(2)
        beta = Constant(9e-3)
        L = (
            # relaxation term
            + (beta*epsilon)/2.*inner(grad(rho),grad(rho))*dx
            + beta/(epsilon*2.)*inner(rho, 1.-rho)*dx

            # Linear elasticity PDE constraint & objective function
            # using antisymmetry of Lagrange multiplier
            - 2.0*mu_lame*(self.k(rho, params))*inner(self.symgrad(u), self.symgrad(u))*dx
            - lmbda_lame*(self.k(rho, params))*inner(tr(self.symgrad(u))*I,tr(self.symgrad(u))*I)*dx
            + 2.0*inner(f, u)*ds(1)
            # volume constraint
            - inner(lmbda, gamma - rho)*dx
            )

        return L

    def cost(self, z, params):
        ds = self.boundary_ds(z.function_space().mesh())
        rho, u, lmbda = split(z)
        (gamma, p, eps, f, mu_lame, lmbda_lame, epsilon) = params
        f = Constant((0,f))
        L = (
             inner(f,u)*ds(1)
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1).sub(1), 0.0, South()),
                DirichletBC(Z.sub(1).sub(0), 0.0, West())]


    def initial_guesses(self, Z, params):
        # compute an initial guess by solving the linear elasticity equation
        # with a uniform distribution of rho = 0.535 over the whole domain
        comm = self.G.mesh().mpi_comm()
        commZ = Z.mesh().mpi_comm()
        ds = self.boundary_ds(Z.mesh())

        print("Computing initial guess.")
        (gamma, p, eps, f, mu_lame, lmbda_lame, epsilon) = params
        f = Constant((0,f))
        V = self.volume_constraint(params)
        rho_guess =Constant(V)
        u_guess = Constant(V)

        u = Function(self.G)
        v = TestFunction(self.G)

        I = Identity(2)
        F = ( 2.0*mu_lame*(self.k(rho_guess, params))*inner(self.symgrad(v), self.symgrad(u))*dx
            + lmbda_lame*(self.k(rho_guess, params))*inner(tr(self.symgrad(u))*I, self.symgrad(v))*dx
            - inner(f,v)*ds(1)
            )

        PETScOptions.set("snes_monitor")
        PETScOptions.set("ksp_type", "preonly")
        PETScOptions.set("pc_type", "lu")
        PETScOptions.set("mat_mumps_icntl_14", "1000")
        solver_params = {"nonlinear_solver": "snes"}
        if PETSc.Sys.getVersion()[0:2] < (3, 9):
            PETScOptions.set("pc_factor_mat_solver_package", "mumps")
        else:
            PETScOptions.set("pc_factor_mat_solver_type", "mumps")


        solve(F == 0, u, self.Gbcs, solver_parameters=solver_params)
        print("Initial guess computed, projecting ...")

        lmbda_guess = Constant(10)
        rho_guess = variable(rho_guess)


        ic = as_vector([rho_guess, u[0], u[1], lmbda_guess])
        z = project(ic, Z, solver_type="mumps") # there HAS to be a better way to do this
        print("Initial guess projected.")
        return [z]

    def number_solutions(self, mu, params):
        if float(mu) < 0.17:
            return 2
        else: return 1


    def solver_parameters(self,mu, branch, task, params):
        (gamma, p, eps, f, mu_lame, lmbda_lame, epsilon) = params

        linesearch = "l2"
        if task == 'ContinuationTask':
            max_it = 100
            damping = 1.0
        elif task == 'DeflationTask':
            linesearch = "basic"
            max_it = 300
            damping = 0.1
        elif task == 'PredictorTask':
            linesearch = "basic"
            max_it = 20
            damping = 1.0

        args = {
               "snes_max_it": max_it,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_atol": 1.0e-9,
               "snes_divergence_tolerance": 1.0,
               "snes_converged_reason": None,
               "snes_monitor": None,
               "snes_linesearch_type": linesearch,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": damping,
               "snes_linesearch_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "mumps",
               "mat_mumps_icntl_24": 1,
               "mat_mumps_icntl_13": 1
               }
        return args

    def k(self, rho, params):
        (gamma, p, _, f, mu_lame, lmbda_lame, epsilon) = params
        eps = 1e-5
        eps = Constant(eps)
        return eps + (1 - eps) * rho**p

    def bounds_vi(self, Z, mu, params):
        inf = 1e100
        lb = interpolate(Constant((0.0,-inf, -inf, -inf)), Z)
        ub = interpolate(Constant((1.0,+inf, +inf, +inf)), Z)
        return (lb, ub)


    def volume_constraint(self, params):
        return params[0]

    def update_mu(self, u, mu, iters, k, k_mu_old, params):

        if float(mu) > 1.0:
            k_mu = 0.5
        elif 0.2 < float(mu) <= 1.0:
            k_mu = 0.9
        else:
            k_mu = 0.95

        theta_mu = 1.05
        next_mu = min(k_mu*mu, mu**theta_mu)
        return next_mu

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return feasibletangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)


if __name__ == "__main__":
    problem = MBBProblem()
    # Young's Modulus and Possion's ratio for steel
    YM = 196.0
    nu = 0.3

    # Lame parameters
    mu_lame    = YM/(2.*(1.+nu))
    lmbda      = YM*nu/((1.+nu)*(1.-2.*nu))
    lmbda_lame = 2.0*lmbda*mu_lame/(lmbda + 2.0*mu_lame)

    epsilon = 0.11313708498984613
    params = [0.535, 3, 0.0, -10.0, mu_lame, lmbda_lame, epsilon] #(gamma, p, eps, f, mu_lame, lmbda_lame)
    deflatedbarrier(problem, params, mu_start=50., mu_end = 1e-10, max_halfstep = 3)

    # (un)comment out below for grid-sequencing and continuation of epsilon parameter
    def parameter_update(epsilon, z):
       return 0.7*epsilon

    params = [0.535, 3, 0.0, -10.0, mu_lame, lmbda_lame, epsilon]
    gridsequencing(problem, sharpness_coefficient = 6, branches = [0],
                   params = params, iters_total = 5,
                   parameter_update = parameter_update,
                   mu_start_continuation = 1e-5, grid_refinement = 2)

    params = [0.535, 3, 0.0, -10.0, mu_lame, lmbda_lame, epsilon]
    gridsequencing(problem, sharpness_coefficient = 6, branches = [1],
                   params = params, iters_total = 5,
                   parameter_update = parameter_update,
                   mu_start_continuation = 1e-5, grid_refinement = 2)
