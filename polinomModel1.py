import numpy as np
import math
from dataset1 import ti, yi

ti = np.array(ti).reshape(-1, 1)
yi = np.array(yi).reshape(-1, 1)

N = ti.shape[0]
J = -np.ones((N, 1))
J = np.hstack((J, -ti))
J = np.hstack((J, -ti ** 2))
A = np.dot(J.transpose(), J)
B = np.dot(J.transpose(), yi)
x = -np.dot(np.linalg.inv(A), B)

#------------------------------------------------------------#

import matplotlib.pyplot as plt

T = np.arange(-3, 3, 0.01)
yhat = x[0] + x[1] * T + x[2] * T ** 2
plt.scatter(ti, yi, color = 'darkred')
plt.plot(T, yhat)
plt.xlabel('ti')
plt.ylabel('yi')
plt.title('Dataset 1', fontstyle = 'italic')
plt.grid(color = 'green', linestyle = '--', linewidth = 0.1)
plt.show()