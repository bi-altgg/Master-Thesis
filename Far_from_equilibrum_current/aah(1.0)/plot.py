import numpy as np
import matplotlib.pyplot as plt
yaxis = np.loadtxt('current_file(0.1)(1.0)(check).txt', dtype = float)
xaxis = np.loadtxt('x_axis(check).txt', dtype = float)
plt.plot(xaxis,yaxis * xaxis)
plt.yscale('log')
plt.xscale('log')
plt.show()