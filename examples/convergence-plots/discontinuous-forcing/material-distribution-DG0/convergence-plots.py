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
comm = MPI.COMM_WORLD

# Throw an error if the saved solutions are not where they should be
if not os.path.isdir("../../deflatedbarrier-data"):
    import deflatedbarrier
    deflatedbarrier.info_red("\nRequire solutions to generate convergence plots.\nPlease run 'make data-download' in the parent directory")
    exit()

def load_mesh(pathfile):
    mesh = Mesh()
    comm = mesh.mpi_comm()
    h5 = HDF5File(comm, pathfile, "r")
    h5.read(mesh, "/mesh", False)
    del h5
    return mesh

comm = MPI.COMM_WORLD

# Load meshes
mesh0 = Mesh("../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/mesh-0.xml")
mesh1 = Mesh("../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/mesh-1.xml")
mesh2 = load_mesh("../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/mesh-2.xml")

# Mixed finite element space
Ve = VectorElement("CG", triangle, 2) # velocity
Pe = FiniteElement("CG", triangle, 1) # pressure
Ce = FiniteElement("DG", triangle, 0) # control
Re = FiniteElement("R",  triangle, 0) # reals
ZeDG = MixedElement([Ce, Ve, Pe, Re, Re])
ZeCG = MixedElement([Pe, Ve, Pe, Re, Re])

# Load refined-mesh solutions to be used a proxies for the anlytical solution
Z_fine_0  = FunctionSpace(mesh0, ZeCG)
z_fine_0= Function(Z_fine_0)
h5 = HDF5File(comm, "../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/0.xml.gz", "r")
h5.read(z_fine_0, "/guess")
del h5

Z_fine_1  = FunctionSpace(mesh1, ZeCG)
z_fine_1= Function(Z_fine_1)
h5 = HDF5File(comm, "../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/1.xml.gz", "r")
h5.read(z_fine_1, "/guess")
del h5

Z_fine_2  = FunctionSpace(mesh2, ZeCG)
z_fine_2= Function(Z_fine_2)
h5 = HDF5File(comm, "../../deflatedbarrier-data/discontinuous-forcing/refined-solutions/2.xml.gz", "r")
h5.read(z_fine_2, "/guess")
del h5

(rho_0, u_0, p_0, _, _) = z_fine_0.split()
(rho_1, u_1, p_1, _, _) = z_fine_1.split()
(rho_2, u_2, p_2, _, _) = z_fine_2.split()


list_rho = [[],[],[]]
list_u_l2 = [[],[],[]]
list_u = [[],[],[]]
list_p = [[],[],[]]
h = []

# Run through coarse mesh solutions and compute the error norms
for N in [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:
   mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), N,N)
   hmin = mesh.hmin()
   h.append(hmin)
   Z  = FunctionSpace(mesh, ZeDG)
   z = Function(Z)
   z_ = Function(Z)
   for branch in range(0,3):
       fol1 = "../../deflatedbarrier-data/discontinuous-forcing/material-distribution-DG0/"
       fol2 = "mu-0.000000000000e+00-hmin-%.3e-params-[0.3333333333333333, 25000.0, 0.1]-solver-BensonMunson/%d.xml.gz"%(hmin,branch)
       h5 = HDF5File(mesh.mpi_comm(), fol1 + fol2, "r")
       h5.read(z, "/guess")
       del h5
       (rho, u, p, _, _) = z.split(True)

       if branch == 0:
           rho_ = rho_0; u_ = u_0; p_ = p_0

       elif branch == 1:
           rho_ = rho_1; u_ = u_1; p_ = p_1

       else:
           rho_ = rho_2; u_ = u_2; p_ = p_2

       # Compute norm
       list_rho[branch].append(errornorm(rho_, rho, norm_type='L2'))
       list_u[branch].append(errornorm(u_,u, norm_type='H1'))
       list_u_l2[branch].append(errornorm(u_,u, norm_type='L2'))
       list_p[branch].append(errornorm(p_,p, norm_type='L2'))
       print("Finished N = %s"%N)

# Sort out the data

rho_0 = list_rho[0]
u_0   = list_u[0]
u_0_l2 = list_u_l2[0]
p_0   = list_p[0]

rho_1 = list_rho[1]
u_1   = list_u[1]
u_1_l2 = list_u_l2[1]
p_1   = list_p[1]

rho_2 = list_rho[2]
u_2   = list_u[2]
u_2_l2 = list_u_l2[2]
p_2   = list_p[2]


hnorm = np.asarray(h)/h[0]

print("h = %s" %h)
print("rho_0 = %s" %rho_0)
print("u_0 = %s" %u_0)
print("u_0_l2 = %s" %u_0_l2)
print("p_0 = %s" %p_0)

print("h = %s" %h)
print("rho_1 = %s" %rho_1)
print("u_1 = %s" %u_1)
print("u_1_l2 = %s" %u_1_l2)
print("p_1 = %s" %p_1)

print("h = %s" %h)
print("rho_2 = %s" %rho_2)
print("u_2 = %s" %u_2)
print("u_2_l2 = %s" %u_2_l2)
print("p_2 = %s" %p_2)

try:
    os.makedirs('figures')
except:
    pass

h = np.asarray(h)
h1 = h


u_0 = np.asarray(u_0)
u_0_l2 = np.asarray(u_0_l2)
rho_0 = np.asarray(rho_0)
p_0 = np.asarray(p_0)

u_1 = np.asarray(u_1)
u_1_l2 = np.asarray(u_1_l2)
rho_1 = np.asarray(rho_1)
p_1 = np.asarray(p_1)

u_2 = np.asarray(u_2)
u_2_l2 = np.asarray(u_2_l2)
rho_2 = np.asarray(rho_2)
p_2 = np.asarray(p_2)

sub01 = range(10)
sub02 = []
u_01 = u_0[sub01]
u_02 = u_0[sub02]
u_l2_01 = u_0_l2[sub01]
u_l2_02 = u_0_l2[sub02]
p_01 = p_0[sub01]
p_02 = p_0[sub02]
rho_01 = rho_0[sub01]
rho_02 = rho_0[sub02]

sub11 = range(10)
sub12 = []
u_11 = u_1[sub11]
u_12 = u_1[sub12]
u_l2_11 = u_1_l2[sub11]
u_l2_12 = u_1_l2[sub12]
p_11 = p_1[sub11]
p_12 = p_1[sub12]
rho_11 = rho_1[sub11]
rho_12 = rho_1[sub12]

sub21 = range(10)
sub22 = []
u_21 = u_1[sub21]
u_22 = u_1[sub22]
u_l2_21 = u_1_l2[sub21]
u_l2_22 = u_1_l2[sub22]
p_21 = p_1[sub21]
p_22 = p_1[sub22]
rho_21 = rho_1[sub21]
rho_22 = rho_1[sub22]

# Plot the convergence on loglog plots
plt.loglog(h[sub11],u_11, color = 'b', marker = 'o', linewidth = 3.5, label = r"Upper annulus")
#plt.loglog(h[sub12],u_12, color = 'b', marker = 'o')
plt.loglog(h[sub01],u_01, color = 'orange', marker = 'x', label = r"Figure eight")
#plt.loglog(h[sub02],u_02, color = 'orange', marker = 'x')
plt.loglog(h[sub21],u_21, color = 'lime', marker = 'o', markersize = 2, label = r"Lower annulus")
#plt.loglog(h[sub22],u_22, color = 'red', marker = 'o')
plt.loglog(h,hnorm*u_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
plt.loglog(h,hnorm**2*u_0[0], color = 'm', linestyle = '--',label = r"$\mathcal{O}(h^2)$")
plt.legend(loc = 0)
plt.title(r"$H^1(\Omega)$-norm error of the velocity", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|u - u_h\|_{H^1(\Omega)}$", fontsize = 20)
plt.xticks([1e-2,1e-1], fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_u.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h[sub11],u_l2_11, color = 'b', marker = 'o', linewidth = 3.5, label = r"Upper annulus")
# plt.loglog(h[sub12],u_l2_12, color = 'b', marker = 'o')
plt.loglog(h[sub01],u_l2_01, color = 'orange', marker = 'x', label = r"Figure eight")
# plt.loglog(h[sub02],u_l2_02, color = 'orange', marker = 'x')
plt.loglog(h[sub21],u_l2_21, color = 'lime', marker = 'o', markersize = 2, label = r"Lower annulus")
# plt.loglog(h[sub22],u_l2_22, color = 'red', marker = 'o')
plt.loglog(h,hnorm*u_0_l2[0], color = 'g', linestyle = '--',label = r"$\mathcal{O}(h)$")
plt.loglog(h,hnorm**2*u_0_l2[0], color = 'm', linestyle = '--',label = r"$\mathcal{O}(h^2)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the velocity", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|u - u_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks([1e-2,1e-1], fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_u_l2.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()

plt.loglog(h[sub11],rho_11, color = 'b', marker = 'o', linewidth = 3.5, label = r"Upper annulus")
# plt.loglog(h[sub12],rho_12, color = 'b', marker = 'o')
plt.loglog(h[sub01],rho_01, color = 'orange', marker = 'x', label = r"Figure eight")
# plt.loglog(h[sub02],rho_02, color = 'orange', marker = 'x')
plt.loglog(h[sub21],rho_21, color = 'lime', marker = 'o', markersize = 2, label = r"Lower annulus")
# plt.loglog(h[sub22],rho_22, color = 'red', marker = 'o')
plt.loglog(h,hnorm*rho_0[0], color = 'g', linestyle = '--', label = r"$\mathcal{O}(h)$")
plt.legend(loc = 0)
plt.title(r"$L^2(\Omega)$-norm error of the material distribution", fontsize = 20)
plt.xlabel(r"$h$", fontsize = 20)
plt.ylabel(r"$\|\rho - \rho_h\|_{L^2(\Omega)}$", fontsize = 20)
plt.xticks([1e-2,1e-1], fontsize = 15)
plt.yticks(fontsize = 15)
plt.savefig('figures/errorplot_rho.pdf', bbox_inches = "tight", pad_inches=0.05)
plt.close()
