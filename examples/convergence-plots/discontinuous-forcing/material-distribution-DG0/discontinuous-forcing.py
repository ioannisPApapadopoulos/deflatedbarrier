# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
from petsc4py import PETSc

"""
Midpoint discontinuous forcing term with zero BCs on the velocity

"""

width = 1.0 # aspect ratio
N = 20  # mesh resolution 

def load_mesh(pathfile):
    mesh = Mesh()
    comm = mesh.mpi_comm()
    h5 = HDF5File(comm, pathfile, "r")
    h5.read(mesh, "/mesh", False)
    del h5
    return mesh

class ForcingTerm(UserExpression):
    """ 
    This class represents a forcing term which is equal to (10,0) 
    between (0.3,0.3) and (0.7,0.7) and equal to (0,0) otherwise

    """
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        midpoint = 0.5
        if 0.3 < x[0] < 0.7 and 0.3 < x[1] < 0.7:
            values[0] = 10.0
    def value_shape(self):
        return (2,)

class BorrvallProblem(PrimalInteriorPoint):
    def mesh(self, comm):
        mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(width, 1.0), int(width*N),N)
        info_green("Mesh size %s" %mesh.hmin())
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", mesh.ufl_cell(), 2) # velocity
        Pe = FiniteElement("CG", mesh.ufl_cell(), 1) # pressure
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0) # control
        Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals

        Ze = MixedElement([Ce, Ve, Pe, Re, Re])
        Z  = FunctionSpace(mesh, Ze)
        self.no_dofs = Z.dim()
        print("Number of degrees of freedom: ", self.no_dofs)

        # Take some data. First, BCs
        self.force = ForcingTerm(element=Ve)
        
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
            - inner(self.force, u)*dx
            )
        return L

    def number_solutions(self, mu, params):
        if float(mu) > 5e-3:
            return 3
        else:
            return 3

    def cost(self, z, params):
        rho, u, p, p0, lmbda = split(z)
        L = (
              0.5 * inner(grad(u), grad(u))*dx
            + 0.5 * self.alpha(rho, params) * inner(u, u)*dx
            -       inner(self.force, u)*dx
            )
        C = assemble(L)
        return C

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(1), Constant((0.0,0.0)), "on_boundary")]


    def initial_guesses(self, Z, params):
        """
        Use as initial guess the constant rho that satisfies the integral constraint.
        Solve the Stokes equations for the values of (u, p, p0).
        """
        comm = self.G.mesh().mpi_comm()
        commZ = Z.mesh().mpi_comm()
        
        mesh0 = Mesh("../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/mesh-0.xml")
        mesh1 = Mesh("../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/mesh-1.xml")
        mesh2 = load_mesh("../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/mesh-2.xml")
        Ve = VectorElement("CG", mesh0.ufl_cell(), 2) # velocity
        Pe = FiniteElement("CG", mesh0.ufl_cell(), 1) # pressure
        Ce = FiniteElement("CG", mesh0.ufl_cell(), 1) # control
        Re = FiniteElement("R",  mesh0.ufl_cell(), 0) # reals
        Ze = MixedElement([Ce, Ve, Pe, Re, Re])
        Z0  = FunctionSpace(mesh0, Ze)
        Z1  = FunctionSpace(mesh1, Ze)
        Z2  = FunctionSpace(mesh2, Ze)
        
        h5 = HDF5File(commZ, "../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/0.xml.gz", "r")
        zf = Function(Z0)
        h5.read(zf, "/guess")
        z0 = Function(Z)
        z0.interpolate(zf)
        
        h5 = HDF5File(commZ, "../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/1.xml.gz", "r")
        zf = Function(Z1)
        h5.read(zf, "/guess")
        z1 = Function(Z)
        z1.interpolate(zf)
        
        h5 = HDF5File(commZ, "../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/2.xml.gz", "r")
        zf = Function(Z2)
        h5.read(zf, "/guess")
        z2 = Function(Z)
        z2.interpolate(zf)
        
        return [z0,z1,z2]
        
    def solver_parameters(self, mu, branch, task, params):
        (gamma,alphabar, q) = params
        if float(mu) < 100:
            linesearch = "l2"
        else:
            linesearch = "basic"
        if task == 'ContinuationTask':
            max_it = 50
            damping = 1.0
        elif task == 'DeflationTask':
            max_it = 200
            damping = 1.0 #0.2 - original that found first two branches
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
            - inner(self.force,u)*dx
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

    def save_pvd(self, pvd, z, mu):
         if float(mu) == 0.0:
            rho_ = z.split(deepcopy=True)[0]
            rho_.rename("Control", "Control")
            pvd << rho_

if __name__ == "__main__":
    problem = BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
    
    # Using Benson-Munson solver
    deflatedbarrier(problem, params, mu_start= 1e-3, mu_end = 1e-10, max_halfstep = 2)
