import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


#1) Generazione dati a forma di mezza luna 
X,y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Normalizzare i dati
X = StandardScaler().fit_transform(X)

#2) Implementazione DBSCAN 
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=labels_dbscan, cmap='plasma')
plt.title("Clustering con DBSCAN (rileva le lune)")
plt.show()

kmeans = KMeans(n_clusters=2, n_init=10)
labels_kmeans = kmeans.fit_predict(X)
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

ax1.scatter(X[:,0], X[:,1], c=labels_kmeans, cmap='viridis')
ax1.set_title("K-Means (Fallisce  con forme non  sferiche)")
ax2.scatter(X[:,0], X[:,1], c=labels_dbscan, cmap='plasma')
ax2.set_title("DBSCAN (successo con forme dense)")
plt.show()

n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"Cluster trovati: {n_clusters}")
print(f"Punti di rumore: {n_noise}")
if n_clusters > 1:
    print(f"Silhouette Score : {silhouette_score(X,labels_dbscan):.3f}")