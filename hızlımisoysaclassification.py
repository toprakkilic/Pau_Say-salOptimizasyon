import numpy as np
import math
from datasetSpiral import ti, yi
from numba import njit

@njit
def exp(x):
    return np.exp(x)

@njit
def tanh(x):
    return np.tanh(x)

@njit
def MISOYSAmodelIO(ti, Wg, bh, Wc, bc):
    S = Wg.shape[0]
    yhat = np.zeros(ti.shape[0])
    for i in range(ti.shape[0]):
        t = ti[i].reshape(-1, 1)
        h = tanh(Wg @ t + bh)
        nn = Wc @ h + bc
        yhat[i] = nn[0, 0]
    return yhat

@njit
def error(Wg, bh, Wc, bc, ti, yi):
    yhat = MISOYSAmodelIO(ti, Wg, bh, Wc, bc)
    return yi - yhat

@njit
def findJacobian(traininginput, Wg, bh, Wc, bc):
    R = traininginput.shape[1]
    S = Wg.shape[0]
    numofdata = len(traininginput)
    J = np.zeros((numofdata, S*(R+2)+1))
    for i in range(numofdata):
        for j in range(S*R):
            k = j % S
            m = j // S
            temp = tanh(Wg[k, :] @ traininginput[i] + bh[k, 0])
            J[i, j] = -Wc[0, k] * traininginput[i, m] * (1 - temp * temp)
        for j in range(S*R, S*R+S):
            temp = tanh(Wg[j - S*R, :] @ traininginput[i] + bh[j - S*R, 0])
            J[i, j] = -Wc[0, j - S*R] * (1 - temp * temp)
        for j in range(S*R+S, S*(R+2)):
            temp = tanh(Wg[j - (R+1)*S, :] @ traininginput[i] + bh[j - (R+1)*S, 0])
            J[i, j] = -temp
        J[i, S*(R+2)] = -1
    return J

def Matrix2Vector(Wg, bh, Wc, bc):
    return np.concatenate([Wg.flatten(), bh.flatten(), Wc.flatten(), bc.flatten()])

def Vector2Matrix(z, S, R):
    Wgz = z[:S*R].reshape(S, R)
    bhz = z[S*R:S*(R+1)].reshape(S, 1)
    Wcz = z[S*(R+1):S*(R+2)].reshape(1, S)
    bcz = z[S*(R+2)].reshape(1, 1)
    return Wgz, bhz, Wcz, bcz

# Veri Hazırlığı
trainingindices = np.arange(0, len(ti), 2)
traininginput = np.array(ti)[trainingindices]
trainingoutput = np.array(yi)[trainingindices]
validationindices = np.arange(1, len(ti), 2)
validationinput = np.array(ti)[validationindices]
validationoutput = np.array(yi)[validationindices]

# Başlangıç Parametreleri
MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99
R = ti.shape[1]
S = 300

Wg = np.random.random((S, R)) - 0.5
bh = np.random.random((S, 1)) - 0.5
Wc = np.random.random((1, S)) - 0.5
bc = np.random.random((1, 1)) - 0.5
xk = Matrix2Vector(Wg, bh, Wc, bc)

k = 0
fvalidationBest = 1e99
C1 = C2 = C3 = C4 = True

ek = error(Wg, bh, Wc, bc, traininginput, trainingoutput)
ftraining = sum(ek**2)
FTRA = [math.log10(ftraining)]
evalidation = error(Wg, bh, Wc, bc, validationinput, validationoutput)
fvalidation = sum(evalidation**2)
FVAL = [math.log10(fvalidation)]
ITERATION = [k]
print('k:', k, ' f:', format(ftraining, 'f'))

mu = 1
muscal = 10
I = np.identity(S*(R+2)+1)

# Ana Döngü
while C1 and C2 and C3 and C4:
    ek = error(Wg, bh, Wc, bc, traininginput, trainingoutput)
    Jk = findJacobian(traininginput, Wg, bh, Wc, bc)
    gk = 2 * Jk.T @ ek.reshape(-1, 1)
    Hk = 2 * Jk.T @ Jk + 1e-8 * I
    ftraining = sum(ek**2)
    sk = 1
    loop = True
    while loop:
        zk = -np.linalg.inv(Hk + mu * I) @ gk
        z = xk + zk.flatten()
        Wgz, bhz, Wcz, bcz = Vector2Matrix(z, S, R)
        ez = error(Wgz, bhz, Wcz, bcz, traininginput, trainingoutput)
        fz = sum(ez**2)
        if fz < ftraining:
            pk = zk.flatten()
            mu = mu / muscal
            k += 1
            xk = xk + sk * pk
            Wg = Wgz
            bh = bhz
            Wc = Wcz
            bc = bcz
            loop = False
            print('k:', k, ' ftra:', format(fz, 'f'), ' fval:', format(fvalidation, 'f'), ' fval*:', format(fvalidationBest, 'f'))
        else:
            mu = mu * muscal
            if mu > mumax:
                loop = False
                C2 = False
    evalidation = error(Wg, bh, Wc, bc, validationinput, validationoutput)
    fvalidation = sum(evalidation**2)
    if fvalidation < fvalidationBest:
        fvalidationBest = fvalidation
        xkbest = xk.copy()
    FTRA.append(math.log10(ftraining))
    FVAL.append(math.log10(fvalidation))
    ITERATION.append(k)

    C1 = k < MaxIter
    C2 = epsilon1 < abs(ftraining - fz)
    C3 = epsilon2 < np.linalg.norm(sk * pk)
    C4 = epsilon3 < np.linalg.norm(gk)
