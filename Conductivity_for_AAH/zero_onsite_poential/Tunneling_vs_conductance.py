#Nearest neighbour hoping
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

Nmst = 201; #Number of lattice points
Pbst = 101;#Probe attachment site
lbd = 0.0 ;#Lambda strength ofAAH
irrb = (1 + np.sqrt(5))/2
to = 3.0; # hopping potential for bath
sitegammaindx = [0, Nmst-1, Pbst-1]
sitegammastrn = [1.0, 1.0, 0.1]
def hmlt(ratio): #Hamiltonain
    t1 = 1.0; # hopping potential for nearest sites
    t2 = ratio* t1 
    siteindx = np.array(range(1, Nmst+1))
    sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
    diagonals = [sitepotential,t1*np.ones(Nmst-1), t1*np.ones(Nmst-1),t2*np.ones(Nmst-2),t2*np.ones(Nmst-2)]
    offset = [0,-1,1,-2,2]
    sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
    return sys_Ham
def selfenergy(gamma,site,energy):#self energy matrix
    mat = np.zeros((Nmst,Nmst),dtype = 'complex_')
    mat[site,site] =  ((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j))
    return mat
def specden(gamma,energy):#spectral density matrix(-2Im(sigma))
    mat =  -2*((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)).imag
    return mat
def ret_gre(energy,sys_Ham):#green's function
    mat = energy*np.identity(Nmst) - sys_Ham - selfenergy(sitegammastrn[0],sitegammaindx[0],energy) - \
    selfenergy(sitegammastrn[1],sitegammaindx[1],energy) - selfenergy(sitegammastrn[2],sitegammaindx[2],energy)
    return (np.linalg.det(mat)/t)
def adv_gre(energy,sys_Ham):
    mat = np.transpose(np.conjugate(Green_func(energy,sys_Ham)))
    return mat
def trnasmission(sgindx1,sgstrn1,sgindx2,sgstrn2,energy, arraysitegamstrn):
    spcdn1 = specden(sgstrn1,sgindx1,energy)
    retgre = ret_gre(energy, arraysitegamstrn)
    spcdn2 = specden(sgstrn2,sgindx2,energy)
    advgre = adv_gre(energy, arraysitegamstrn)
    mat = np.dot(np.dot(spcdn1,retgre),np.dot(spcdn2,advgre))
    return np.trace(mat)
Rthop = np.logspace(-2,1,200)
for i in range(len(Rthop)):
    sys_Ham = hmlt(Rthop[i])

