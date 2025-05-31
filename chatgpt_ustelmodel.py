import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_data = np.array([-4.0, -3.2, -2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4, 3.2])  
y_data = np.array([-0.3, -0.49, -0.8, -1.31, -2.13, -3.47, -5.66, -9.23, -15.04, -24.53]) 

# 2. Model tanımı
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# 3. Katsayıları bulma (a ve b)
params, _ = curve_fit(exponential_model, x_data, y_data)
a, b = params
print(f"Tahmin edilen a = {a:.4f}, b = {b:.4f}")

# 4. Grafikle gösterim
x_line = np.linspace(min(x_data), max(x_data), 100)
y_line = exponential_model(x_line, a, b)

plt.scatter(x_data, y_data, label="Gerçek Veri")
plt.plot(x_line, y_line, color="red", label="Tahmin Edilen Üstel Model")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Doğrulama: Üstel Model")
plt.grid(True)
plt.show()
