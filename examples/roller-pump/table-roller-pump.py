# -*- coding: utf-8 -*-
from dolfin import *
from deflatedbarrier import *
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

rollerpump = __import__("roller-pump")

def create_table_BM():
    problem = rollerpump.RollerPumpProblem()
    params = [1.0/3, 2.5e4, 0.1] #(gamma, alphabar, q)
    table = np.array([[r'$h_{\mathrm{min}}$/$h_{\mathrm{max}}$', r'Dofs', r'Cont.', r'Defl.', r'Pred.', r'Cont.', r'Defl.', r'Pred.']])

    for mesh_type in ['verycoarse', 'coarse', 'fine']:

        class RollerPump(object):
            def mesh(self, comm):
                mesh = Mesh("mesh/%s.xml"%mesh_type)
                # Mark boundaries of pump and domain
                sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
                sub_domains.set_all(0)
                right = rollerpump.Dirichlet()
                right.mark(sub_domains, 1)
                right = rollerpump.DirichletRoller()
                right.mark(sub_domains, 2)

                self.hmin = mesh.hmin()
                self.hmax = mesh.hmax()
                return mesh

            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = RollerPump()
        (_, out) = deflatedbarrier(newproblem, params, mu_end = 1e-5)
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
    plt.xlabel(r"BM solver iterations for the roller-pump problem", fontsize = 30)
    plt.savefig("table.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close()

create_table_BM()
