#AAH critical
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
from tqdm import tqdm
#Creating the spectral density matrix
n=11;#laatice sites
no=6;#bath lattice point
b = (1+np.sqrt(5))/2
sitegammaindx = [0, n-1,no-1]
gammastrn = np.logspace(-2,3,1000)
arrayofsitegamstrn = []
for i in gammastrn:
    sitegammastrn = [1.0,1.0,i]
    arrayofsitegamstrn += [sitegammastrn]
to = 3.0 #bath tunneling potential
t = 1.0 #system hopping
lamba = 1.0
alpha = 0.0
siteindx = np.array(range(1, n+1))
sitepotential = 2*lamba*np.cos(2*np.pi*b*(siteindx))
diagonals = [sitepotential,t*np.ones(n-1), t*np.ones(n-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
#sitepotential = 0.0;#bath site potential(constant)
siteindx = np.array(range(1, n+1))
sitepotentialAAH = 2*lamba*np.cos(2*np.pi*b*(siteindx))/(1+alpha*np.cos(2*np.pi*b*(siteindx)))


def selfenergy(gamma,site,energy):#self energy matrix
    mat = np.zeros((n,n),dtype = 'complex_')
    mat[site,site] =  ((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j))
    return mat
def specden(gamma,site,energy):#spectral density matrix(-2Im(sigma))
    mat = np.zeros((n,n),dtype = 'complex_')
    mat[site,site] =   -2*(((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)).imag)
    return mat
#Green's functions
def ret_gre(energy,arraysitgamstrn):
    mat = energy*np.identity(n) - sys_Ham - selfenergy(arraysitgamstrn[0],sitegammaindx[0],energy) - \
    selfenergy(arraysitgamstrn[1],sitegammaindx[1],energy) - selfenergy(arraysitgamstrn[2],sitegammaindx[2],energy)    
    return (np.linalg.inv(mat)/t)


def adv_gre(energy,arraysitgamstrn):
    return np.transpose(np.conjugate(ret_gre(energy,arraysitgamstrn)))

#transmission probability
def trnasmission(sgindx1,sgstrn1,sgindx2,sgstrn2,energy,arraysitgamstrn):
    spcdn1 = specden(sgstrn1,sgindx1,energy)
    retgre = ret_gre(energy,arraysitgamstrn)
    spcdn2 = specden(sgstrn2,sgindx2,energy)
    advgre = adv_gre(energy,arraysitgamstrn)
    mat = np.dot(np.dot(spcdn1,retgre),np.dot(spcdn2,advgre))
    return np.trace(mat)


free_energy = np.loadtxt('(1.0)free_energ11.txt',dtype = 'float')
eigval = np.loadtxt('eigenvalue(1.0)11.dat',dtype = 'float')
mat = []
for sgstrn in tqdm(arrayofsitegamstrn):
    fe = eigval[10]#eigval
    print(sgstrn,fe)
    rl = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    nr = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[1],sgstrn[1],fe, sgstrn).real
    nl = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    rn = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[2],sgstrn[2],fe, sgstrn).real
    if nr + nl == 0:
        mat += [rl]
    else:
        mat += [(rl+((rn*nl)/(nr+nl)))]
plot = [m if m>1.0E-18 else 1.0E-18 for m in mat]
np.savetxt('new(1.0)conductivity v_s strength_at11'+'%1.2f'%fe+'.txt',plot)
mat=[]
for sgstrn in tqdm(arrayofsitegamstrn):
    fe = eigval[5]#eigval
    print(sgstrn,fe)
    rl = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    nr = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[1],sgstrn[1],fe, sgstrn).real
    nl = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    rn = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[2],sgstrn[2],fe, sgstrn).real
    if nr + nl == 0:
        mat += [rl]
    else:
        mat += [(rl+((rn*nl)/(nr+nl)))]
plot = [m if m>1.0E-18 else 1.0E-18 for m in mat]
np.savetxt('new(1.0)conductivity v_s strength_at11'+'%1.2f'%fe+'.txt',plot)
mat=[]
for sgstrn in tqdm(arrayofsitegamstrn):
    fe = free_energy[401]
    print(sgstrn,fe)
    rl = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    nr = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[1],sgstrn[1],fe, sgstrn).real
    nl = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    rn = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[2],sgstrn[2],fe, sgstrn).real
    if nr + nl == 0:
        mat += [rl]
    else:
        mat += [(rl+((rn*nl)/(nr+nl)))]
plot = [m if m>1.0E-18 else 1.0E-18 for m in mat]
np.savetxt('new(1.0)conductivity v_s strength_at11'+'%1.2f'%fe+'.txt',plot)
mat=[]
for sgstrn in tqdm(arrayofsitegamstrn):
    fe = 1.40
    print(sgstrn,fe)
    rl = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    nr = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[1],sgstrn[1],fe, sgstrn).real
    nl = trnasmission(sitegammaindx[2],sgstrn[2],sitegammaindx[0],sgstrn[0],fe, sgstrn).real
    rn = trnasmission(sitegammaindx[1],sgstrn[1],sitegammaindx[2],sgstrn[2],fe, sgstrn).real
    if nr + nl == 0:
        mat += [rl]
    else:
        mat += [(rl+((rn*nl)/(nr+nl)))]
plot = [m if m>1.0E-18 else 1.0E-18 for m in mat]
np.savetxt('new(1.0)conductivity v_s strength_at11'+'%1.2f'%fe+'.txt',plot)
np.savetxt('x-axis.txt',gammastrn)
plt.grid()
plt.yscale('log')
plt.xscale('log')



