import numpy as np
import scipy.linalg as la
import cmath,math
from scipy.sparse import diags
from scipy.integrate import quad
from scipy import integrate
from scipy import optimize
from scipy.misc import derivative
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

Nmst = 200; #Number of lattice points
Pbst = 100;#Probe attachment site
lbd = 0.0 ;#Lambda strength ofAAH
irrb = (1 + np.sqrt(5))/2
t = 1.0; # hopping potential for sites
to = 3.0; # hopping potential for bath
sitegammaindx = [0, Nmst-1,Pbst - 1]
sitegammastrn = [1.0, 1.0, 0.0]
siteindx = np.array(range(1, Nmst+1))
sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
diagonals = [sitepotential,t*np.ones(Nmst-1), t*np.ones(Nmst-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
del_mu = 1.0e-6
def giveratio(x,y,z):
    if x==y:
        poi =  500
    else:
        poi = math.ceil(abs((y-x)/(z-x)))*500
    return poi

mu_L = 0.02 - (del_mu/2)#left chemical potential
mu_R = 0.02 + (del_mu/2) #right chemical potential(first value, last value, no. of values)
print(mu_R)
def selfenergy(gamma,site,energy):#self energy matrix
    mat = np.zeros((Nmst,Nmst),dtype = 'complex_')
    mat[site,site] =  ((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j))
    return mat
def specden(gamma,site,energy):#spectral density matrix(-2Im(sigma))
    mat = np.zeros((Nmst,Nmst),dtype = 'complex_')
    mat[site,site] =   -2*(((gamma**2/(2*to**2))*(energy - cmath.sqrt(4*to**2-energy**2)*1j)).imag)
    return mat
#Green's functions
def ret_gre(energy):
    mat = energy*np.identity(Nmst) - sys_Ham - selfenergy(sitegammastrn[0],sitegammaindx[0],energy) - \
    selfenergy(sitegammastrn[1],sitegammaindx[1],energy) - selfenergy(sitegammastrn[2],sitegammaindx[2],energy)    
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
def current_fulllr(mu_l, mu_r):
    global parrelel_trnas3
    point = np.linspace(mu_l,mu_r,1000)   
    def parrelel_trnas3(x):
        return trnasmission(sitegammaindx[1],sitegammastrn[1],sitegammaindx[0],sitegammastrn[0],x).real 
    result = list(map(parrelel_trnas3,point))
    first_trans_right = result
    plt.plot(point,result)
    plt.show()
    return first_trans_right
def current_full2(mu_l,mu_r):  
    point1 = np.linspace(mu_l,mu_r,1000) 
    I1 = current_fulllr(mu_l,mu_r)
    return(integrate.simps(I1,point1))
print(current_full2(mu_L,mu_R)/(mu_R-mu_L))
current_array = []

"""
for i in range(len(del_mu)): 
    current_array.append((current_full2(mu_L,mu_R[i])))
    print(current_array)
    print('done',i)
plt.plot(del_mu,current_array)
plt.show()
"""