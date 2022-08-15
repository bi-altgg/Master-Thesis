#zero_onsite
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

SMALL_SIZE = 25
MEDIUM_SIZE = 25
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
#Creating the spectral density matrix
n=200;#laatice sites
no=100;#bath lattice point
b = (1+np.sqrt(5))/2
sitegammaindx = [0, n-1, no-1]
sitegamaastrn = [1.0,1.0,5.0]
to = 3.0 #bath tunneling potential
t = 1.0 #system hopping
lamba = 0.0
alpha = 0.0
#sitepotential = 0.0;#bath site potential(constant)
siteindx = np.array(range(1, n+1))
sitepotentialAAH = 2*lamba*np.cos(2*np.pi*b*(siteindx))
diagonals = [sitepotentialAAH,t*np.ones(n-1), t*np.ones(n-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
eigvals, eigvecs = la.eig(sys_Ham)
energyval = eigvals.real
def selfenergy(gamma,site,energy):#self energy matrix
    mat = np.zeros((n,n),dtype = 'complex_')
    mat[site,site] =  ((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j))
    return mat
def specden(gamma,site,energy):#spectral density matrix(-2Im(sigma))
    mat = np.zeros((n,n),dtype = 'complex_')
    mat[site,site] =  -2*(((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)).imag)
    return mat
#Green's functions
def ret_gre(energy):
    mat = energy*np.identity(n) - sys_Ham - selfenergy(sitegamaastrn[0],sitegammaindx[0],energy) - \
    selfenergy(sitegamaastrn[1],sitegammaindx[1],energy) - selfenergy(sitegamaastrn[2],sitegammaindx[2],energy)    
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



fe = 0.02
rl = trnasmission(sitegammaindx[1],sitegamaastrn[1],sitegammaindx[0],sitegamaastrn[0],fe).real
nr = trnasmission(sitegammaindx[2],sitegamaastrn[2],sitegammaindx[1],sitegamaastrn[1],fe).real
nl = trnasmission(sitegammaindx[2],sitegamaastrn[2],sitegammaindx[0],sitegamaastrn[0],fe).real
rn = trnasmission(sitegammaindx[1],sitegamaastrn[1],sitegammaindx[2],sitegamaastrn[2],fe).real
if nr + nl == 0:
    mat = rl
else:
    mat = (rl+(rn*nl)/(nr+nl))
print(mat)


