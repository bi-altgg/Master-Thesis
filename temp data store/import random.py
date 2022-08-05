import random
from re import I
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
Pbst = 100;#Probe attachment site
lbd = 0.0 ;#Lambda strength ofAAH
irrb = (1 + np.sqrt(5))/2
t = 1.0; # hopping potential for sites
to = 3.0; # hopping potential for bath
sitegammaindx = [0, Nmst-1]
sitegammastrn = [1.0, 1.0]
siteindx = np.array(range(1, Nmst+1))
sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
diagonals = [sitepotential,t*np.ones(Nmst-1), t*np.ones(Nmst-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
mu_L = 0.02 #left chemical potential
mu_R = 0.02 + 1.0e-6 #right chemical potential(first value, last value, no. of values)
beta_left = 1000
beta_right = 1000
beta_probe = 1000
def bosonic_distribution(mu, energy, beta):#bosonic distribution function
    mat = []
    for i in range(len(energy)):
        mat.append((np.exp(beta*(energy[i] - mu))+1)**(-1))
    return np.array(mat)
def selfenergy(gamma,energy):
    mat = ((gamma**2)/(2*to**2))*(energy - np.sqrt(4*to**2-energy**2)*1j)
    return mat
def specden(gamma,site,energy):#spectral density matrix(-2Im(sigma))
    mat = np.zeros((Nmst,Nmst),dtype = 'complex_')
    mat[site,site] =  -2*(selfenergy(gamma,energy).imag)
    return mat
#Green's functions
def ret_gre(energy):
    k = [-t*np.ones(Nmst-1),np.ones(Nmst),-t*np.ones(Nmst-1)]
    offset = [-1,0,1]
    mat = diags(k,offset,dtype='complex_').toarray()
    for i in range(Nmst):
        mat[i, i] = (energy - sitepotential[i]) / t
    for i in range(2):
        mat[sitegammaindx[i],sitegammaindx[i]] = (energy - sitepotential[sitegammaindx[i]] - selfenergy(sitegammastrn[i],energy))/t
    return (np.linalg.inv(mat)/t)


def adv_gre(energy):
    return np.transpose(np.conjugate(ret_gre(energy)))

#transmission probability
def trnasmission(sgindx1,sgstrn1,sgindx2,sgstrn2,energy):
    spcdn1 = specden(sgstrn1,sgindx1,energy)
    retgre = ret_gre(energy)
    spcdn2 = specden(sgstrn2,sgindx2,energy)
    advgre = adv_gre(energy)
    mat = np.dot(np.dot(spcdn1,retgre),np.dot(spcdn2,advgre))
    return np.trace(mat)
point = np.linspace(-3.0,3.0,4000)
transmissionprob = np.vectorize(trnasmission)
first_trans_right = transmissionprob(sitegammaindx[1],sitegammastrn[1], sitegammaindx[0],sitegammastrn[0],point).real

def current_full(mu_l, mu_r):
    left_bath = bosonic_distribution(mu_l, point, beta_left)
    right_bath = bosonic_distribution(mu_r, point, beta_right)
    return (first_trans_right*(right_bath - left_bath))
def current_full2(mu_l,mu_r):  
    I = current_full(mu_l, mu_r)
    return(integrate.simps(I,point))

print((current_full2(mu_L,mu_R))/(mu_R-mu_L))
