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
sitegammaindx = [0, Nmst-1, Pbst-1]
sitegammastrn = [1.0, 1.0, 0.05]
siteindx = np.array(range(1, Nmst+1))
sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
diagonals = [sitepotential,t*np.ones(Nmst-1), t*np.ones(Nmst-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
mu_L = 0.2 #left chemical potential
mu_R = 0.2 + np.logspace(-6,-2,200) #right chemical potential(first value, last value, no. of values)
beta_left = 100
beta_right = 100
beta_probe = 100
def bosonic_distribution(mu, energy, beta):#bosonic distribution function
    mat = []
    for i in range(len(energy)):
        mat.append((np.exp(beta*(energy[i] - mu))+1)**(-1))
    return np.array(mat)
def selfenergy(gamma,site,energy):#self energy matrix
    mat = np.zeros((Nmst,Nmst),dtype = 'complex_')
    mat[site,site] =  ((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j))
    return mat
def specden(gamma,energy):#spectral density matrix(-2Im(sigma))
    mat =  -2*((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)).imag
    return mat
def specden2(gamma,site , energy):#spectral density matrix(-2Im(sigma))
    mat = np.zeros((Nmst,Nmst),dtype = 'complex_')
    mat[site,site] =  -2*((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)).imag
    return mat
def Green_func(energy):#green's function
    mat = energy*np.identity(Nmst) - sys_Ham - selfenergy(sitegammastrn[0],sitegammaindx[0],energy) - \
    selfenergy(sitegammastrn[1],sitegammaindx[1],energy) - selfenergy(sitegammastrn[2],sitegammaindx[2],energy)
    return (np.linalg.inv(mat)/t)
def advan_gren(energy):
    mat = np.transpose(np.conjugate(Green_func(energy)))
    return mat
def transmissionprob(sgstrn1,sgnindx1, sgstrn2, sgnindx2, energy):
    spcdn1 = specden2(sgstrn1,sgnindx1,energy)
    retgre = Green_func(energy)
    spcdn2 = specden2(sgstrn2, sgnindx2 ,energy)
    advgre = advan_gren(energy)
    mat = np.dot(np.dot(spcdn1,retgre),np.dot(spcdn2,advgre))
    return np.trace(mat)
point = np.linspace(-3.0,3.0,2000)
transmissionprob = np.vectorize(transmissionprob)
first_trans_probe = transmissionprob(sitegammastrn[2],sitegammaindx[2], sitegammastrn[0],sitegammaindx[0],point).real
second_trans_probe = transmissionprob(sitegammastrn[2],sitegammaindx[2], sitegammastrn[1],sitegammaindx[1],point).real
first_trans_right = transmissionprob(sitegammastrn[1],sitegammaindx[1], sitegammastrn[0],sitegammaindx[0],point).real
second_trans_right = transmissionprob(sitegammastrn[1],sitegammaindx[1], sitegammastrn[2],sitegammaindx[2],point).real
print(second_trans_probe)

def current_val(mu_p,mu_l,mu_r):#integral equation
    left_bath = bosonic_distribution(mu_l, point, beta_left)
    probe = bosonic_distribution(mu_p, point, beta_probe)
    right_bath = bosonic_distribution(mu_r, point, beta_right)
    return (first_trans_probe*(probe - left_bath)) + (second_trans_probe*(probe - right_bath))
def current_int(mu_p,mu_l,mu_r):
    I = current_val(mu_p,mu_l,mu_r)
    print('i')
    print(len(I),len(point))
    return(integrate.simps(I,point))
def deriv_current(mu_p,mu_l,mu_r) :
    return derivative(lambda x: current_int(x,mu_l,mu_r),mu_p)
def min_probe_potential(mu_l,mu_r):#Newton-Raphson minimization
    Mu = (mu_r+mu_l)/2
    current =  current_int(Mu,mu_l,mu_r)
    while abs(current) >= 1.0e-10:
        Mu = Mu - (current_int(Mu,mu_l,mu_r)/(deriv_current(Mu,mu_l,mu_r)))
        current =  current_int(Mu,mu_l,mu_r)
        print(current)
    return Mu
def current_full(mu_l, mu_r):
    mu_poi = min_probe_potential(mu_l,mu_r)
    left_bath = bosonic_distribution(mu_l, point, beta_left)
    probe = bosonic_distribution(mu_poi, point, beta_probe)
    right_bath = bosonic_distribution(mu_r, point, beta_right)
    print(len(first_trans_right),len(right_bath),len(left_bath))
    return (first_trans_right*(right_bath - left_bath) + second_trans_right*(right_bath - probe))
def current_full2(mu_l,mu_r):  
    I = current_full(mu_l, mu_r)
    return(integrate.simps(I,point))
plot = []
for i in range(len(mu_R)):
    plot.append((current_full2(mu_L,mu_R[i]))/mu_R[i])
np.savetxt("/home/bishal/Master's thesis/Codes/Far_from_equilibrum_current/aah(1.0)/current_file(0.1)(1.0)(check2)" + '.txt', plot )
np.savetxt("/home/bishal/Master's thesis/Codes/Far_from_equilibrum_current/aah(1.0)/x_axis(check).txt", mu_R)
plt.plot(mu_R,plot)
print('j')
plt.show()