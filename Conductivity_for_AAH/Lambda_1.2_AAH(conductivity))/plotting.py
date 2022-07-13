import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
sgstrn = [0.0,0.1,0.5,1.0,2.0,5.0,10.0]
for i in sgstrn:
    plot = np.loadtxt('datafile_for'+ str(i) + '.txt', dtype = 'float')
    free_energy = np.loadtxt('free_energ.txt',dtype = 'float')
    plt.plot(free_energy, plot,'o',label =  f'$\gamma = {i}$')

plt.title('$G/G_o$ v/s $\epsilon_{F}$ for various probe strength ($\lambda = 0.5$)')
plt.xlabel('$\epsilon_{F}$')
plt.ylabel('$G/G_o$')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()