import numpy as np
import math

# ornekFonksiyon2.py dosyasındaki f ve gradient fonksiyonlarının kullanılması
from ornekFonksiyon2 import f
from ornekFonksiyon2 import gradient as gradf

# Altın kesim yöntemi (line search)
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

# Conjugate Gradient (CG) Algoritması
x = np.array([2, 2])  # Başlangıç noktası
X1 = [x[0]]
X2 = [x[1]]
Nmax = 1000  # Maksimum iterasyon sayısı
eps1 = 1e-10  # Fonksiyon değeri değişim toleransı
eps2 = 1e-10  # Değişkenler değişim toleransı
eps3 = 1e-10  # Gradyan toleransı
k = 0

# Başlangıçtaki updatedx değeri
updatedx = np.array([1e10, 1e10])

# Durma koşulları
C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

# İlk adım için başlangıç
r = gradf(x)  # İlk gradyan
p = -r  # İlk konjugat yön (p0 = -gradyan)
x_Before = x  # Önceki x
pk_Before = p  # Önceki p

# İterasyon döngüsü
while not (C1 or C2 or C3 or C4):
    grad = gradf(x)  # Gradyanı hesapla

    if k != 0:
        # Beta'yı hesapla
        beta_Ust = np.dot(grad.T, grad)  # r_k^T * r_k
        beta_Alt = np.dot(gradf(x_Before).T, gradf(x_Before))  # r_{k-1}^T * r_{k-1}
        beta = beta_Ust / beta_Alt  # Beta hesaplama
    
        p = -grad + beta * pk_Before  # Yeni konjugat yönünü hesapla
    
    # Adım boyutunu hesapla (line search)
    #sk = GSmain(f, x, p)  # Altın kesim yöntemiyle öğrenme oranını hesapla
    sk = 0.001
    x_Before = x
    x = x + sk * p  # x'i güncelle
    pk_Before = p  # Önceki p'yi tutuyoruz
    
    # Durma koşullarını tekrar kontrol et
    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3
    
    updatedx = 1*x  # updatedx'i x ile güncelle
    X1.append(x[0])
    X2.append(x[1])

    k += 1
    print(f"Iterasyon {k}: x = [{x[0]:.4f}, {x[1]:.4f}], f(x) = {f(x):.4f}")

# Sonuçları yazdır
if C1:
    print("... MAX ITERASYON SAYISINA ULASILDI ...")
if C2:
    print("... FONKSIYON DEGISMIYOR ...")
if C3:
    print("... DEGISKENLER DEGISMIYOR ...")
if C4:
    print("... DURAGAN NOKTAYA GELINDI ...")
