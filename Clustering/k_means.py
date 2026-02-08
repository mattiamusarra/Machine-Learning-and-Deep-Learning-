import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Impostiamo lo stile estetico 
sns.set_theme(style="whitegrid")

#1) Generazione dati( 4 gruppi distanti)
X,y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#2) Inizializzazione e addestramento del modello n_init = 10 quanto volte riparte con c diverso
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

plt.scatter(X[:,0],X[:,1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, alpha=0.7, marker='X', label='Centroidi')
plt.title("Risultato K-Means con 4 Cluster")
plt.legend()
plt.show()

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i, n_init=10, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11), wcss, marker='o')
plt.title("Metodo del Gomito (Elbow Method)")
plt.xlabel("Numero di Cluster")
plt.ylabel('WCSS (Inertia)')
plt.show()

score = silhouette_score(X, y_kmeans)
print(f"Silhoutte Score per k=4: {score:.3f}")