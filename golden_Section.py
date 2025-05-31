import math

def GSmain(f, xk, pk):
    xalt = 0
    xust = 1
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xust - xalt)
    N = round(-2.078 * math.log(epsilon))
    
    k = 0
    x1 = xalt + tau * (xust - xalt)
    f1 = f(xk + x1 * pk)
    x2 = xust - tau * (xust - xalt)
    f2 = f(xk + x2 * pk)
    
    for k in range(0, N):
        if f1 > f2:
            xalt = x1
            x1 = x2
            f1 = f2
            x2 = xust - tau * (xust - xalt)
            f2 = f(xk + x2 * pk)
        else:
            xust = x2
            x2 = x1
            f2 = f1
            x1 = xalt + tau * (xust - xalt)
            f1 = f(xk + x1 * pk)
            
    x = 0.5 * (x1 + x2)
    return x
