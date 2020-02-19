# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

doublepipe = __import__("neumann-outlet-double-pipe")

"""
This script generates a table of the number of BM solver iterations required to solve
the neumann-outlet-double-pipe problem for varying refinements of meshes

"""
width = 1.5
def create_table_BM():
    problem = doublepipe.BorrvallProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
    table = np.array([[r'$h$', r'Dofs', r'Cont.', r'Defl.', r'Pred.', r'Cont.', r'Defl.', r'Pred.', r'Cont.', r'Defl.', r'Pred.', r'Cont.', r'Defl.', r'Pred.']])

    for cell_no in [30, 40, 80]:

        class DoublePipe(object):
            def mesh(self, comm):
                mesh = RectangleMesh(comm, Point(0.0, 0.0), Point(width, 1.0), int(width*cell_no),cell_no, "crossed")
                self.hmin = mesh.hmin()
                return  mesh
            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = DoublePipe()
        (_, out) = deflatedbarrier(newproblem, params, mu_end = 1e-5)
        if cell_no == 30:
            # for cell_no 30, branch 2 is discovered before branch 1
            branch1cont = out[0,3]
            branch1defl = out[0,4]
            branch1pred = out[0,5]
            out[0,3] = out[0,6]
            out[0,4] = out[0,7]
            out[0,5] = out[0,8]
            out[0,6] = branch1cont
            out[0,7] = branch1defl
            out[0,8] = branch1pred
        out = np.append(["%.4f"%(newproblem.hmin),"%d"%newproblem.no_dofs], out)
        table = np.append(table, [out], axis = 0)

    fig, ax = plt.subplots()
    columns = (r'BM solver', '', r'Branch 0', r'Branch 0', r'Branch 0', r'Branch 1', r'Branch 1', r'Branch 1', r'Branch 2', r'Branch 2', r'Branch 2', r'Branch 3', r'Branch 3', r'Branch 3')
    the_table = plt.table(cellText=table, colLabels=columns, loc='top')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(6, 4)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.xlabel(r"BM solver iterations for the double-pipe problem with Neumann boundary conditions on the outlet", fontsize = 30)
    plt.savefig("table.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

create_table_BM()
