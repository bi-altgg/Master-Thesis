
from re import I
import numpy as np
import scipy.linalg as la
import cmath
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
sitegammaindx = [0, Nmst-1,Pbst - 1]
sitegammastrn = [1.0, 1.0, 0.5]
siteindx = np.array(range(1, Nmst+1))
sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
diagonals = [sitepotential,t*np.ones(Nmst-1), t*np.ones(Nmst-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
mu_L = 0.02 #left chemical potential
mu_R = 0.02 + 1.0e-6 #right chemical potential(first value, last value, no. of values)
print(mu_R)
def selfenergy(gamma,energy):
    mat = ((gamma**2)/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)
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
    for i in range(3):
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
trnasmission = np.vectorize(trnasmission)
def current_vallp(mu_p,mu_l):#integral equation
    point = np.linspace(mu_l,mu_p,150)
    first_trans_probe = trnasmission(sitegammaindx[2],sitegammastrn[2],sitegammaindx[0],sitegammastrn[0],point).real
    print('j')
    return first_trans_probe
def current_valpr(mu_p,mu_r):
    point = np.linspace(mu_r,mu_p,150)
    second_trans_probe = trnasmission(sitegammaindx[2],sitegammastrn[2],sitegammaindx[1],sitegammastrn[1],point).real
    print('k')
    return second_trans_probe
def current_int(mu_p,mu_l,mu_r):
    point1 = np.linspace(mu_l,mu_p,150)
    I1 = current_vallp(mu_p,mu_l)
    point2 = np.linspace(mu_r,mu_p,150)
    I2 = current_valpr(mu_p,mu_r)
    return(integrate.simps(I1,point1) + integrate.simps(I2,point2))
def deriv_current(mu_p,mu_l,mu_r) :
    return derivative(lambda x: current_int(x,mu_l,mu_r),mu_p)
def min_probe_potential(mu_l,mu_r):#Newton-Raphson minimization
    Mu = (mu_r+mu_l)/2
    current =  current_int(Mu,mu_l,mu_r)
    while abs(current) >= 1.0e-10:
        Mu = Mu - (current_int(Mu,mu_l,mu_r)/(deriv_current(Mu,mu_l,mu_r)))
        current =  current_int(Mu,mu_l,mu_r)
        print('i')
    return Mu
def current_fulllr(mu_l, mu_r):
    point = np.linspace(mu_l,mu_r,150)    
    first_trans_right = trnasmission(sitegammaindx[1],sitegammastrn[1],sitegammaindx[0],sitegammastrn[0],point).real
    print('m')
    return first_trans_right
def current_fulllp(mu_r, mu_l):
    mu_poi = min_probe_potential(mu_l,mu_r)
    point = np.linspace(mu_poi,mu_r,150) 
    second_trans_right = trnasmission(sitegammaindx[1],sitegammastrn[1],sitegammaindx[2],sitegammastrn[2],point).real
    print('n')
    return second_trans_right
def current_full2(mu_l,mu_r):  
    mu_poi = min_probe_potential(mu_l,mu_r)
    point1 = np.linspace(mu_l,mu_r,150) 
    I1 = current_fulllr(mu_l,mu_r)
    point2 = np.linspace(mu_poi,mu_r,150) 
    I2 = current_fulllp(mu_r, mu_l)
    print('o')
    return(integrate.simps(I1,point1) + integrate.simps(I2,point2))

print((current_full2(mu_L,mu_R))/abs(mu_R-mu_L))