import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import math,cmath
from scipy.sparse import diags
sgstrn = [0.0,0.1,0.5,1.0,2.0,5.0,10.0]
colormark =[['green',"."],['blue',"."],['red',"."],['tab:grey',"."],['tab:cyan',"."],['tab:pink',"."],['tab:olive',"."]]
for i in range(7):
    plot = np.loadtxt('datafile_for'+ str(sgstrn[i]) + '.txt', dtype = 'float')
    free_energy = np.loadtxt('free_energ.txt',dtype = 'float')
    plt.scatter(free_energy, plot,c = colormark[i][0],marker = colormark[i][1],alpha = .8,label =  f'$\gamma = {sgstrn[i]}$')

plt.title('$G/G_o$ v/s $\epsilon_{F}$ for various probe strength ($\lambda = 0.5$)')
plt.xlabel('$\epsilon_{F}$')
plt.ylabel('$G/G_o$')
plt.yscale('log')
plt.grid()
plt.legend()
plt.ylim((1.0E-17,1.02))
plt.xlim((-2.5,+2.5))
plt.show()
