import numpy as np
import math

from ornekFonksiyon2 import f, hessian
from ornekFonksiyon2 import gradient as gradf

def GSmain(f, xk, pk):
    xalt = 0
    xust = 1
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xust - xalt)
    N = round(-2.078 * math.log(epsilon))
    
    k = 0
    x1 = xalt + tau * (xust - xalt); f1 = f(xk + x1 * pk)
    x2 = xust - tau * (xust - xalt); f2 = f(xk + x2 * pk)
    
    for k in range(0, N):
        
        if f1 > f2:
            xalt = 1 * x1; x1 = 1 * x2; f1 = 1 * f2
            x2 = xust - tau * (xust - xalt); f2 = f(xk + x2 * pk)
        else:
            xust = 1 * x2; x2 = 1 * x1; f2 = 1 * f1
            x1 = xalt + tau * (xust - xalt); f1 = f(xk + x1 * pk)
            
    x = 0.5 * (x1 + x2)
    return x


#--------------------------------------------------------------------#

x = np.array([-5.4, 1.7])
X1 = [x[0]]
X2 = [x[1]]
Nmax = 10000
eps1 = 1e-10
eps2 = 1e-10
eps3 = 1e-10
k = 0

#--------------------------------------------------------------------#

I = np.identity(2)
M = np.identity(2)
updatedx = np.array([1e10, 1e10])
C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

#--------------------------------------------------------------------#

while not (C1 or C2 or C3 or C4):
    k += 1
    ozdeger, ozvektor = np.linalg.eig(M)
    
    if np.min(ozdeger) > 0:
        pk = -np.dot(np.linalg.inv(M), gradf(x))
    else:
        mu = abs(np.min(ozdeger)) + 0.001
        pk = -np.dot((np.linalg.inv(M + mu * I)), gradf(x))
        
    sk = GSmain(f, x, pk)
    prevG = gradf(x).reshape(-1, 1)
    x = x + sk * pk
    x = np.array(x)
    currentG = gradf(x).reshape(-1, 1)
    y = (currentG - prevG)
    pk = pk.reshape(-1, 1)
    Dx = (sk * pk)
    
    A = np.dot(y, np.matrix.transpose(y)) / np.dot(np.matrix.transpose(y), Dx)
    B = np.dot(np.dot(M, Dx), np.dot(np.matrix.transpose(Dx), M)) / np.dot(np.matrix.transpose(Dx), np.dot(M, Dx))
    M = M + A - B
    print("k: ", k, "sk: ", round(sk, 4), "x1: ", round(x[0], 4), "x2: ", round(x[1], 4), "f: ", round(f(x), 4), "||f(x)||: ", round(np.linalg.norm(gradf(x)), 4))
    
    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3
    
    updatedx = 1 * x
    X1.append(x[0])
    X2.append(x[1])

#--------------------------------------------------------------------#

if C1:
    print("... MAX ITERASYON SAYISINA ULASILDI ...")
if C2:
    print("... FONKSIYON DEGISMIYOR ...")
if C3:
    print("... DEGISKENLER DEGISMIYOR ...")
if C4:
    print("... DURAGAN NOKTAYA GELINDI ...")

#--------------------------------------------------------------------#

import matplotlib.pyplot as plt
plt.plot(X1, X2)
plt.scatter(X1, X2, s = 5, c = 'red')
plt.show()