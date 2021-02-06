# -*- coding: utf-8 -*-

from dolfin import *
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
import numpy as np
import os

if not os.path.isdir("../../deflatedbarrier-data"):
    import deflatedbarrier
    deflatedbarrier.info_red("\nRequire solutions to generate convergence plots.\nPlease run 'make data-download' in the parent directory")
    exit()

comm = MPI.COMM_WORLD

delta = 1.5
mesh0 = Mesh("../../deflatedbarrier-data/double-pipe/refined-solutions/mesh-0.xml")
mesh1 = Mesh("../../deflatedbarrier-data/double-pipe/refined-solutions/mesh-1.xml")
Ve = VectorElement("CG", triangle, 2) # velocity
Pe = FiniteElement("CG", triangle, 1) # pressure
Ce = FiniteElement("CG", triangle, 1) # control
Re = FiniteElement("R",  triangle, 0) # reals

Ze = MixedElement([Ce, Ve, Pe, Re, Re])
Z_fine_0  = FunctionSpace(mesh0, Ze)
Z_fine_1  = FunctionSpace(mesh1, Ze)
z_fine_0= Function(Z_fine_0)
z_fine_1= Function(Z_fine_1)
h5 = HDF5File(comm, "../../deflatedbarrier-data/double-pipe/refined-solutions/0.xml.gz", "r")
h5.read(z_fine_0, "/guess")
del h5
h5 = HDF5File(comm, "../../deflatedbarrier-data/double-pipe/refined-solutions/1.xml.gz", "r")
h5.read(z_fine_1, "/guess")
del h5
(rho_0, u_0, p_0, _, _) = z_fine_0.split()
(rho_1, u_1, p_1, _, _) = z_fine_1.split()

list_rho = [[],[]]
list_u = [[],[]]
list_u_l2 = [[],[]]
list_p = [[],[]]
h = []

for N in [25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]: # removed 80 and 105, 90 and  100

    mesh = Mesh("../../deflatedbarrier-data/double-pipe/unstructured-mesh/mesh/N-%s/mesh.xml"%N)
    hmin = mesh.hmin()
    h.append(hmin)

    Z  = FunctionSpace(mesh, Ze)
    z = Function(Z)
    z_ = Function(Z)

    for branch in range(0,2):
        fol1 = "../../deflatedbarrier-data/double-pipe/unstructured-mesh/N-%s-output/"%N
        fol2 = "mu-0.000000000000e+00-hmin-%.3e-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/%d.xml.gz"%(hmin,branch)
        h5 = HDF5File(mesh.mpi_comm(), fol1 + fol2, "r")
        h5.read(z, "/guess")
        del h5
        (rho, u, p, _, _) = z.split(True)
        if branch == 0:
            rho_ = rho_0; u_ = u_0; p_ = p_0
        else:
            rho_ = rho_1; u_ = u_1; p_ = p_1

        # Compute error norm
        list_rho[branch].append(errornorm(rho_, rho, norm_type='L2'))
        list_u[branch].append(errornorm(u_,u, norm_type='H1'))
        list_u_l2[branch].append(errornorm(u_,u, norm_type='L2'))
        list_p[branch].append(errornorm(p_,p, norm_type='L2'))
        print("Finished N = %s, branch = %s"%(N,branch))

rho_0 = list_rho[0]
rho_1 = list_rho[1]
u_0   = list_u[0]
u_1   = list_u[1]
p_0   = list_p[0]
p_1   = list_p[1]
u_0_l2 = list_u_l2[0]
u_1_l2 = list_u_l2[1]

hnorm = np.asarray(h)/h[0]
print("h = %s" %h)
print("rho_0 = %s" %rho_0)
print("rho_1 = %s" %rho_1)
print("u_0 = %s" %u_0)
print("u_1 = %s" %u_1)
print("u_0_l2 = %s" %u_0_l2)
print("u_1_l2 = %s" %u_1_l2)
print("p_0 = %s" %p_0)
print("p_1 = %s" %p_1)

try:
    os.makedirs('figures')
except:
    pass

h = np.asarray(h)

u_0 = np.asarray(u_0)
u_0_l2 = np.asarray(u_0_l2)
rho_0 = np.asarray(rho_0)
p_0 = np.asarray(p_0)

u_1 = np.asarray(u_1)
u_1_l2 = np.asarray(u_1_l2)
rho_1 = np.asarray(rho_1)
p_1 = np.asarray(p_1)

sub01 = [0,1,2,4,5,6,7,9,10]
sub02 = []
u_01 = u_0[sub01]
u_02 = u_0[sub02]
u_l2_01 = u_0_l2[sub01]
u_l2_02 = u_0_l2[sub02]
p_01 = p_0[sub01]
p_02 = p_0[sub02]
rho_01 = rho_0[sub01]
rho_02 = rho_0[sub02]

sub11 = [0,1,3,4,6,8,10] 
sub12 = []
u_11 = u_1[sub11]
u_12 = u_1[sub12]
u_l2_11 = u_1_l2[sub11]
u_l2_12 = u_1_l2[sub12]
p_11 = p_1[sub11]
p_12 = p_1[sub12]
rho_11 = rho_1[sub11]
rho_12 = rho_1[sub12]


plt.loglog(h[sub01],u_01, color = 'b', marker = 'x', label = "Straight channels")
plt.loglog(h[sub12],u_12,color = 'orange', marker = 'o', label = "Double-ended wrench")
plt.loglog(h[sub11],u_11,color = 'orange', marker = 'o')
#plt.loglog(h,hnorm*u_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
#plt.loglog(h,hnorm**1.5*u_0[0], color = 'r', linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*u_0[0], color = 'm', linestyle = '--',label = r"$\mathcal{O}(h^2)$")
plt.legend(loc = 0)
plt.title(r"$H^1(\Omega)$-norm error of the velocity", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|u - u_h\|_{H^1(\Omega)}$", fontsize = 20)
plt.xticks([5e-2,1e-2], fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_u.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h[sub01],u_l2_01, color = 'b', marker = 'x', label = "Straight channels")
plt.loglog(h[sub12],u_l2_12, color = 'orange', marker = 'o', label = "Double-ended wrench")
plt.loglog(h[sub11],u_l2_11, color = 'orange', marker = 'o')
#plt.loglog(h,hnorm*u_0_l2[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
# plt.loglog(h,hnorm**1.5*u_0_l2[0], linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*u_l2_01[0], color = 'm', linestyle = '--',label = r"$\mathcal{O}(h^2)$")
#plt.loglog(h,hnorm**3*u_l2_01[0], color = 'y', linestyle = '--',label = r"$\mathcal{O}(h^3)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the velocity", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|u - u_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks([5e-2,1e-2], fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_u_l2.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h[sub01],rho_01, color = 'b', marker = 'x', label = "Straight channels")
plt.loglog(h[sub12],rho_12, color = 'orange', marker = 'o', label = "Double-ended wrench")
plt.loglog(h[sub11],rho_11, color = 'orange', marker = 'o')
#plt.loglog(h,hnorm*rho_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
#plt.loglog(h,hnorm**1.5*rho_0[0], color = 'r', linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*rho_0[0], color = 'm', linestyle = '--', label = r"$\mathcal{O}(h^2)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the material distribution", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|\rho - \rho_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks([5e-2,1e-2], fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_rho.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h[sub01],p_01, color = 'b', marker = 'x', label = "Straight channels")
plt.loglog(h[sub12],p_12, color = 'orange', marker = 'o', label = "Double-ended wrench")
plt.loglog(h[sub11],p_11, color = 'orange', marker = 'o')
#plt.loglog(h,hnorm*p_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
#plt.loglog(h,hnorm**1.5*p_0[0], color = 'r', linestyle = '--',label = r"$\mathcal{O}(h^{3/2})$")
#plt.loglog(h,hnorm**2*p_0[0], color = 'm', linestyle = '--', label = r"$\mathcal{O}(h^2)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the pressure", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|p - p_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks([5e-2,1e-2], fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_p.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()
