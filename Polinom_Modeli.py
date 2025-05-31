import numpy as np
import matplotlib.pyplot as plt

# Örnek veri
t = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
y = np.array([-1.2557, -0.1074, 0.6040, 0.9605, 1.0730, 1.1060, 1.4090, 1.0223, 0.7559])

# Polinom derecesi (n-1). Örn: 2. derece polinom için n=3
n = 9

def polynomial_model(t, x):
    return np.array([sum([x[j] * (ti ** j) for j in range(n)]) for ti in t])

def error(x, t, y):
    yhat = polynomial_model(t, x)
    return y - yhat

def jacobian(t, x):
    J = np.zeros((len(t), n))
    for i in range(len(t)):
        for j in range(n):
            J[i, j] = -t[i] ** j
    return J

def gauss_newton(t, y, x_init, iterations=10):
    x = np.array(x_init, dtype=float)
    for _ in range(iterations):
        e = error(x, t, y).reshape(-1, 1)
        J = jacobian(t, x)
        delta = np.linalg.pinv(J.T @ J) @ J.T @ e
        x = x - delta.ravel()
    return x

# Başlangıç tahmini
x0 = np.ones(n)

# Polinom katsayılarını hesapla
x_final = gauss_newton(t, y, x0)

# Katsayıları yazdır
print(f"\n📌 Polinom katsayıları (derece {n-1}):")
for i, coef in enumerate(x_final):
    print(f"  x{i} = {coef:.6f}")

# Tahminler
y_pred = polynomial_model(t, x_final)

# Grafik
t_plot = np.linspace(min(t), max(t), 200)
plt.plot(t, y, 'ro', label="Gerçek Veri")
plt.plot(t_plot, polynomial_model(t_plot, x_final), 'b-', label=f"Model (derece {n-1})")
plt.title("Polinom Modeli")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
