import numpy as np
import scipy.linalg as la
import cmath,math
from scipy.sparse import diags
from scipy.integrate import quad
from scipy import integrate
from scipy import optimize
from scipy.misc import derivative
from concurrent.futures import ProcessPoolExecutor
import sys
import time

start = time.time()
Nmst = 5; #Number of lattice points
Pbst = 3;#Probe attachment site
lbd = 0.0 ;#Lambda strength ofAAH
irrb = (1 + np.sqrt(5))/2
t = 1.0; # hopping potential for sites
to = 3.0; # hopping potential for bath
sitegammaindx = [0, Nmst-1,Pbst - 1]
sitegammastrn = [1.0, 1.0, 1.0]
siteindx = np.array(range(1, Nmst+1))
sitepotential = 2*lbd*np.cos(2*np.pi*irrb*(siteindx))
diagonals = [sitepotential,t*np.ones(Nmst-1), t*np.ones(Nmst-1)]
offset = [0,-1,1]
sys_Ham = diags(diagonals,offset,dtype='complex_').toarray()
del_mu = np.linspace(1.0e-6,6.0,3)
mu_L = 0.02 #left chemical potential
mu_R = 0.02 + del_mu #right chemical potential(first value, last value, no. of values)
print(mu_R)

def giveratio(x,y,z):
    if x==y:
        poi =  500
    else:
        poi = math.ceil(abs((y-x)/(z-x)))*500
    return poi


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





def parrelel_trnas1(x):
    return trnasmission(sitegammaindx[2],sitegammastrn[2],sitegammaindx[0],sitegammastrn[0],x).real
def parrelel_trnas2(x):
    return trnasmission(sitegammaindx[2],sitegammastrn[2],sitegammaindx[1],sitegammastrn[1],x).real

def integralsum(X, Y):
    return 0 if len(X) == 1 else np.sum(Y) - (Y[0] + Y[-1]) / 2

numbins = 10**4 # tolerance in mu_p of the order of (mu_r - mu_l)/numbins
maxniter = 1000

def min_probe_potential(mu_l, mu_r): # custom Newton-Raphson minimization
    X = np.linspace(mu_l, mu_r, num=numbins+1, endpoint=True)
    Y1 = parrelel_trnas1(X)
    Y2 = parrelel_trnas2(X)
    i = len(X) // 2 # initial guess
    niter = 0
    while True:
        niter += 1
        if (niter > maxniter):
            print('Error: Newton-Raphson minimization failed, exiting!')
            sys.exit(1)

        # derivative at i (twice the value of the function)
        if i == 0:
            der = (Y1[i+1] + Y1[i] + Y2[i+1] + Y2[i]) / 2
        elif i == numbins:
            der = (Y1[i] + Y1[i-1] + Y2[i] + Y2[i-1]) / 2
        else:
            der = (Y1[i+1] + 2*Y1[i] + Y1[i-1] + Y2[i+1] + 2*Y2[i] + Y2[i-1]) / 4

        # actually current times binwidth is the actual current
        current = integralsum(X[:i+1], Y1[:i+1]) - integralsum(X[i:], Y2[i:])

        if der == 0:
            i = 0 if current > 0 else numbins
            continue

        step = round(current / der)
        #print(i, current, der, step)
        if (step == 0):
            break
        i -= step
        if i < 0:
            i = 0
        elif i > numbins:
            i = numbins
    return X[i]





def current_fulllr(mu_l, mu_r):
    global parrelel_trnas3
    point = np.linspace(mu_l,mu_r,giveratio(mu_L+del_mu[0],mu_r,mu_L+del_mu[1]))   
    def parrelel_trnas3(x):
        return trnasmission(sitegammaindx[1],sitegammastrn[1],sitegammaindx[0],sitegammastrn[0],x).real 
    result = list(map(parrelel_trnas3,point))
    first_trans_right = result
    return first_trans_right
def current_fulllp(mu_r, mu_l):
    mu_poi = min_probe_potential(mu_l,mu_r)
    global parrelel_trnas4
    point = np.linspace(mu_poi,mu_r,giveratio(mu_L+del_mu[0],mu_r,mu_L+del_mu[1])) 
    def parrelel_trnas4(x):
        return trnasmission(sitegammaindx[1],sitegammastrn[1],sitegammaindx[2],sitegammastrn[2],x).real  
    result = list(map(parrelel_trnas4,point))
    second_trans_right = result   
    return second_trans_right
def current_full2(mu_l,mu_r):  
    mu_poi = min_probe_potential(mu_l,mu_r)
    point1 = np.linspace(mu_l,mu_r,giveratio(mu_L+del_mu[0],mu_r,mu_L+del_mu[1])) 
    I1 = current_fulllr(mu_l,mu_r)
    point2 = np.linspace(mu_poi,mu_r,giveratio(mu_L+del_mu[0],mu_r,mu_L+del_mu[1])) 
    I2 = current_fulllp(mu_r, mu_l)
    return(integrate.simps(I1,point1) + integrate.simps(I2,point2))
current_array = []
for i in range(len(del_mu)): 
    current_array.append((current_full2(mu_L,mu_R[i])))
    print('done',i)
    print(current_array)
end = time.time()
print(end - start)