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
                def __getattr__(self, attr):
                    return getattr(problem, attr)

            newproblem = DoublePipe()
            (_, out) = deflatedbarrier(newproblem, solver = "HintermullerItoKunisch", params, mu_start= 100, mu_end = 1e-5, max_halfstep = 0)
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
        plt.savefig("table_HIK.pdf", bbox_inches='tight', pad_inches=0.05)
        plt.close()

create_table_BM()
create_table_HIK()
