# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
from petsc4py import PETSc

doublepipe = __import__("double-pipe")

"""
This script generates a table of the number of BM & HIK solver iterations required
to solve the double-pipe problem for varying refinements of meshes

"""
def create_table_BM():
    problem = doublepipe.BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
    table = np.array([[r'$h$', r'Dofs', r'Cont.', r'Defl.', r'Pred.', r'Cont.', r'Defl.', r'Pred.']])

    for cell_no in [50, 80, 100]:

        class DoublePipe(object):
            def mesh(self, comm):
                mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(1.5, 1.0), int(1.5*cell_no),cell_no)
                self.hmin = mesh.hmin()
                return  mesh
            def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
                return tangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)
            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = DoublePipe()
        (_, out) = deflatedbarrier(newproblem, params, mu_start= 100, mu_end = 1e-5, max_halfstep = 0)
        out = np.append(["%.4f"%newproblem.hmin,"%d"%newproblem.no_dofs], out)
        table = np.append(table, [out], axis = 0)

    fig, ax = plt.subplots()
    columns = (r'BM solver', '', r'Branch 0', r'Branch 0', r'Branch 0', r'Branch 1', r'Branch 1', r'Branch 1')
    the_table = plt.table(cellText=table, colLabels=columns, loc='top')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.xlabel(r"BM solver iterations for the double-pipe problem", fontsize = 30)
    plt.savefig("table_BM.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

def create_table_HIK():
    if PETSc.Sys.getVersion()[0:2] < (3, 10):
        # Need PETSc version with working linesearch
        pass
    else:
        problem = doublepipe.BorrvallProblem()
        params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
        table = np.array([[r'$h$', r'Dofs', r'Cont.', r'Defl.', r'Pred.', r'Cont.', r'Defl.', r'Pred.']])

        for cell_no in [50, 80, 100]:

            class DoublePipe(object):
                def mesh(self, comm):
                    mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(1.5, 1.0), int(1.5*cell_no),cell_no)
                    self.hmin = mesh.hmin()
                    return  mesh

                def solver_parameters(self, mu, branch, task, params):
                    (gamma, alphabar, q) = params

                    linesearch = "l2"

                    if task == 'ContinuationTask':
                        max_it = 500
                        damping = 1.0
                    elif task == 'DeflationTask':
                        max_it = 100
                        damping = 0.95
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
                           "snes_linesearch_monitor": None,
                           "ksp_type": "preonly",
                           "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps",
                           "mat_mumps_icntl_24": 1,
                           "mat_mumps_icntl_13": 1,
                           "mat_mumps_icntl_14": 1000,
                           }
                    return args

                def update_mu(self, u, mu, iters, k, k_mu_old, params):
                    k_mu = 0.8
                    theta_mu = 1.15
                    next_mu = min(k_mu*mu, mu**theta_mu)
                    return next_mu

                def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
                    return tangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)

                def __getattr__(self, attr):
                    return getattr(problem, attr)

            newproblem = DoublePipe()
            (_, out) = deflatedbarrier(newproblem, params, solver = "HintermullerItoKunisch", mu_start= 105, mu_end = 1e-5, max_halfstep = 1)
            out = np.append(["%.4f"%newproblem.hmin,"%d"%newproblem.no_dofs], out)
            table = np.append(table, [out], axis = 0)

        fig, ax = plt.subplots()
        columns = (r'BM solver', '', r'Branch 0', r'Branch 0', r'Branch 0', r'Branch 1', r'Branch 1', r'Branch 1')
        the_table = plt.table(cellText=table, colLabels=columns, loc='top')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(24)
        the_table.scale(4, 4)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        for pos in ['right','top','bottom','left']:
            plt.gca().spines[pos].set_visible(False)
        plt.xlabel(r"HIK solver iterations for the double-pipe problem", fontsize = 30)
        plt.savefig("table_HIK.pdf", bbox_inches='tight', pad_inches=0.05)
        plt.close()

def create_table_BM_SV():
    problem = doublepipe.BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
    table = np.array([[r'$h_{\mathrm{min}}$/$h_{\mathrm{max}}$', r'Dofs', r'Cont.', r'Defl.', r'Pred.', r'Cont.', r'Defl.', r'Pred.']])

    for mesh_type in ["coarse", "fine"]:

        class DoublePipe(object):
            def mesh(self, comm):
                mesh = Mesh("mesh/%s.xml"%mesh_type)
                self.hmin = mesh.hmin()
                self.hmax = mesh.hmax()
                return  mesh

            def function_space(self, mesh):
                Ve = VectorElement("CG", mesh.ufl_cell(), 2) # velocity
                Pe = FiniteElement("DG", mesh.ufl_cell(), 1) # pressure
                Ce = FiniteElement("CG", mesh.ufl_cell(), 1) # control
                Re = FiniteElement("R",  mesh.ufl_cell(), 0) # reals

                Ze = MixedElement([Ce, Ve, Pe, Re, Re])
                Z  = FunctionSpace(mesh, Ze)
                self.no_dofs = Z.dim()
                print("Number of degrees of freedom: ", self.no_dofs)

                # Take some data. First, BCs
                doublepipe.BorrvallProblem.expr = doublepipe.InflowOutflow(element=Ve)
                # Next, a function space we use to solve for our initial guess
                Ge = MixedElement([Ve, Pe, Re])
                doublepipe.BorrvallProblem.G = FunctionSpace(mesh, Ge)
                doublepipe.BorrvallProblem.P = FunctionSpace(mesh, Pe)
                return Z

            def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
                return tangent(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)

            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = DoublePipe()
        (_, out) = deflatedbarrier(newproblem, params, mu_start= 100, mu_end = 1e-5, max_halfstep = 0)
        out = np.append(["%.4f/%.4f"%(newproblem.hmin,newproblem.hmax),"%d"%newproblem.no_dofs], out)
        table = np.append(table, [out], axis = 0)

    fig, ax = plt.subplots()
    columns = (r'BM solver', '', r'Branch 0', r'Branch 0', r'Branch 0', r'Branch 1', r'Branch 1', r'Branch 1')
    the_table = plt.table(cellText=table, colLabels=columns, loc='top')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.xlabel(r"BM solver iterations for the double-pipe problem with Scott-Vogelius finite elements", fontsize = 30)
    plt.savefig("table_BM_SV.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

create_table_BM()
create_table_HIK()
create_table_BM_SV()
