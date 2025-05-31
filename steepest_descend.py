import numpy as np
import math

# ornekFonksiyon2.py dosyasındaki f ve gradient fonksiyonlarının kullanılması
# from ornekFonksiyon2 import f
# from ornekFonksiyon2 import gradient as gradf

def f(x):
    
    f= ((x[0] - 2*x[1] + x[2] + 1)**2) + ((x[0] + x[1] - x[2] + 3)**2) + (((-2)*x[0] + x[1] - x[2])**2)
    return f


def gradf(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    
    gk = np.array([(12*x1 - 6*x2 + 4*x3 + 8) , (-6 * x1 + 12*x2 -8*x3 + 2), (4*x1 - 8*x2 + 6*x3 -4) ])
    return gk
    

# Altın kesim yöntemi
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

# Steepest Descent algoritması
x = np.array([-1, 2, 4])  # Başlangıç noktası
X1 = [x[0]]
X2 = [x[1]]
X3 = [x[2]]
Nmax = 10000000  # Maksimum iterasyon sayısı
eps1 = 1e-10  # Fonksiyon değeri değişim toleransı
eps2 = 1e-10  # Değişkenler değişim toleransı
eps3 = 1e-10  # Gradyan toleransı
k = 0

# Başlangıçtaki updatedx değeri
updatedx = np.array([1e10, 1e10,1e10])

# Durma koşulları
C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

# İterasyon döngüsü
while not (C1 or C2 or C3 or C4):
    k += 1
    grad = gradf(x)  # Gradyanı hesapla
    pk = -grad  # Gradyan yönünde adım at
    #sk = GSmain(f, x, pk)  # Altın kesim yöntemiyle öğrenme oranını hesapla
    sk = 0.01
    x = x + sk * pk  # x'i güncelle

    # Her iterasyonda x değerini yazdır
    print(f"Iterasyon {k}: x = [{x[0]}, {x[1]}, {x[2]}] , f(x) = {f(x):}")


    # Durma koşullarını tekrar kontrol et
    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3
    
    updatedx = 1*x  # updatedx'i x ile güncelle
    X1.append(x[0])
    X2.append(x[1])
    X3.append(x[2])

# Sonuçları yazdır
if C1:
    print("... MAX ITERASYON SAYISINA ULASILDI ...")
if C2:
    print("... FONKSIYON DEGISMIYOR ...")
if C3:
    print("... DEGISKENLER DEGISMIYOR ...")
if C4:
    print("... DURAGAN NOKTAYA GELINDI ...")
