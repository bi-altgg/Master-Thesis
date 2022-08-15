import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
from scipy.integrate import quad
from mpmath import *
from scipy import integrate
from scipy import optimize
from scipy.misc import derivative

Nmst = 200; #Number of lattice points
Pbst = np.linspace(1,Nmst-1,Nmst-2).astype('int');#bath lattice point
lbd = 0.5 ;#Lambda strength ofAAH
irrb = (1 + np.sqrt(5))/2
t = 1.0; # hopping potential for sites
to = 3.0; # hopping potential for bath
sitegammaindx = [0, Nmst-1]
sitegammastrn = [1.0, 1.0, 1.0]
sitegammastrn0 = [1.0,1.0,0.0]
sitegammastrn0_1 = [1.0,1.0,0.1]
sitegamaastrn0_5 = [1.0,1.0,0.5]
sitegammastrn1_0 = [1.0, 1.0, 1.0]
sitegamaastrn2_0 = [1.0,1.0,2.0]
sitegamaastrn5_0 = [1.0,1.0,5.0]
sitegamaastrn10_0 = [1.0,1.0,10.0]
arrayofsitegamstrn = [sitegammastrn0, sitegammastrn0_1, sitegamaastrn0_5, sitegammastrn1_0,sitegamaastrn2_0,sitegamaastrn5_0,sitegamaastrn10_0]
siteindx = np.array(range(1, Nmst+1))
sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
diagonals = [sitepotential,t*np.ones(Nmst-1), t*np.ones(Nmst-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
def selfenergy(gamma,site,energy):#self energy matrix
    mat = np.zeros((Nmst,Nmst),dtype = 'complex_')
    mat[site,site] =  ((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j))
    return mat
def specden(gamma,energy):#spectral density matrix(-2Im(sigma))
    mat =  -2*((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)).imag
    return mat
def Green_func(energy,site):#green's function
    mat = energy*np.identity(Nmst) - sys_Ham - selfenergy(sitegammastrn[0],sitegammaindx[0],energy) - \
    selfenergy(sitegammastrn[1],sitegammaindx[1],energy) - selfenergy(sitegammastrn[2],site,energy)
    return (np.linalg.det(mat)/t)
def advan_gren(energy):
    mat = np.transpose(np.conjugate(Green_func(energy)))
    return mat
def transmissionprob(sitstrn1, sitstrn2, energy,site):
    spcdn1 = specden(sitstrn1, energy)
    retgre = Green_func(energy,site)
    spcdn2 = specden(sitstrn2, energy)
    mat = (spcdn1*spcdn2)/(abs(retgre)**2)
    return mat

free_energy = 0.32
mat = []
for i in Pbst:
    rl = transmissionprob(sitegammastrn[1], sitegammastrn[0],free_energy,i)
    rn = transmissionprob(sitegammastrn[1], sitegammastrn[0],free_energy,i)
    nl = transmissionprob(sitegammastrn[1], sitegammastrn[0],free_energy,i)
    nr = transmissionprob(sitegammastrn[1], sitegammastrn[0],free_energy,i)
    nl = transmissionprob(sitegammastrn[1], sitegammastrn[0],free_energy,i)
    mat.append(rl+(rn*nl)/(nr+nl))

plt.plot(Pbst,mat)
plt.show()