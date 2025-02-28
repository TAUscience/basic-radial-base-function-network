from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from data import X_train, X_test, y_train, y_test

kmeans = KMeans(n_clusters=5)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

def calculo_sigma(centers):
    suma_distancias = 0
    contador = 0
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j:
                suma_distancias += np.linalg.norm(centers[i] - centers[j])
                contador += 1
    return suma_distancias / contador

sigma = calculo_sigma(centers)

def rbf(x, center, sigma):
    return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

def apply_rbf(X, centers, sigma):
    G = np.zeros((X.shape[0], len(centers)))
    for i in range(X.shape[0]):
        for j in range(len(centers)):
            G[i, j] = rbf(X[i], centers[j], sigma)
    return G


rbf_train = apply_rbf(X_train, centers, sigma)
rbf_test = apply_rbf(X_test, centers, sigma)

model = LogisticRegression()
model.fit(rbf_train, y_train)

y_pred = model.predict(rbf_test)

print("Etiquetas clasificadas", y_pred)
print("Etiquetas reales", y_test)

exactitud = accuracy_score(y_test, y_pred)
print("Exactitud del modelo:", exactitud)

