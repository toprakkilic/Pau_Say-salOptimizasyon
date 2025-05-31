import math 

def f(x):
    f = (x-1)**2 * (x-2) *(x-3)
    return f

xalt = 1
xust = 10
dx = 0.0000000000000001
alpha = (1 + math.sqrt(5)) / 2
tau = 1 - 1/alpha
epsilon = dx / (xust + xalt)
N = round(-2.078 *math.log(epsilon)) #adım sayısını bulur 


for k in range (0,N):
    
    x1 = xalt + tau*(xust - xalt); f1 = f(x1)
    x2 = xust - tau *(xust - xalt); f2 = f(x2)

    if(f1 < f2):
        xalt = x1
        x1 = x2
        f1 = f2 
        x2 = xust - tau*(xust-xalt) 
        f2 = f(x2)
        
    else:
        xust = x2
        x2 = x1
        f2 = f1
        x1 = xalt + tau*(xust-xalt) 
        f1 = f(x1)
        
        
    print(k+1,x1,x2,f1,f2)
        

x = (x1 + x2) /f2  #•ikisinin ortalamasını almak daha mantıklı ikisinden birini seçmek yerine
print(x)
