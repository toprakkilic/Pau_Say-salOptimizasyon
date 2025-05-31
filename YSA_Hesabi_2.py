import numpy as np

def tanh(x):
    """Hiperbolik tanjant aktivasyon fonksiyonu."""
    return np.tanh(x)

def ysa_output(t_i, W_g, b_h, W_c, b_c):
    """
    t_i: (R,) boyutunda giriş vektörü
    W_g: (S, R) boyutunda giriş ağırlık matrisi
    b_h: (S,) boyutunda gizli katman bias vektörü
    W_c: (1, S) boyutunda çıkış ağırlık matrisi
    b_c: (1,) boyutunda çıkış bias skalar değer
    """
    hidden_input = np.dot(W_g, t_i) + b_h         # (S,)
    hidden_output = tanh(hidden_input)            # (S,)
    final_output = np.dot(W_c, hidden_output) + b_c  # (1,)
    return final_output[0]  # skalar çıktı

# Örnek parametreler
R = 6  # giriş sayısı
S = 11  # nöron sayısı


t_i = np.array([2.5, 3.0])  

W_g = np.array([
    [-1.4, 0.7],
    [2.3, -0.5,]
])

b_h = np.array([1.3, 0.9])
W_c = np.array([[-0.1, -0.3]])
b_c = np.array([0.5])


# Hesaplama
parametre_Sayisi = S * (R + 2) + 1

output = ysa_output(t_i, W_g, b_h, W_c, b_c)
print(f"Model Çıktısı: {output}")
print(f"Parametre sayısı: {parametre_Sayisi}")

