import numpy as np
import matplotlib.pyplot as plt

# Gaussian RBF fonksiyonu
def gaussian_rbf(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * sigma ** 2))

# RBF ağı
class RBFNetwork:
    def __init__(self, K, sigma=1.0):
        self.K = K              # Merkez sayısı (nöron sayısı)
        self.sigma = sigma
        self.centers = None     # K x özellik sayısı
        self.weights = None     # K x 1

    def _basis_function_matrix(self, X):
        G = np.zeros((X.shape[0], self.K))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = gaussian_rbf(x, c, self.sigma)
        return G

    def fit(self, X, y):
        # Merkezleri rastgele seç
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        self.centers = X[indices]

        # RBF matrisini oluştur
        G = self._basis_function_matrix(X)

        # Ağırlıkları öğren (pseudo-inverse ile)
        self.weights = np.linalg.pinv(G).dot(y)

    def predict(self, X):
        G = self._basis_function_matrix(X)
        return G.dot(self.weights)

# ---------------------
# Örnek kullanım ve grafik
# ---------------------

# Dataset (örnek)
x_data = np.linspace(0, 10, 20).reshape(-1, 1)
y_data = np.sin(x_data).ravel() + 0.1 * np.random.randn(len(x_data))  # Gürültülü sinüs verisi

# RBF ağı oluştur ve eğit
model = RBFNetwork(K=15, sigma=1.0)
model.fit(x_data, y_data)

# Tahmin yapılacak aralık
x_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred = model.predict(x_test)

# Grafik çizimi
plt.scatter(x_data, y_data, color='red', label='Veri Noktaları')
plt.plot(x_test, y_pred, color='blue', label='RBF Tahmini')
plt.title('RBF Ağı ile Veri Uydurma')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
