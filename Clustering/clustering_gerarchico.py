import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import make_blobs 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet

#Crezione dei. dati 
X,_ = make_blobs(n_samples=50, centers=3, cluster_std=0.60, random_state=42)

#Calcolo del legame (Linkage)
#Ward minimizza la varianza dei cluster che vengono fusi
Z = linkage(X, method='ward')

plt.figure(figsize=(10,5))
dendrogram(Z)
plt.title("Dendogramma Gerarchico")
plt.xlabel("Indice del campione")
plt.ylabel("Distanza di Ward")
plt.show()

#Inizializza il modello 
model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')

#Fit e Predict 
labels = model.fit_predict(X)

#Visualizzazione dei risultati 
plt.scatter(X[:,0], X[:,1], c=labels ,cmap='viridis', edgecolors='k')
plt.show()

score = silhouette_score(X,labels)
print(f"Silhouette Score: {score:.2f}")
c, coph_dists = cophenet(Z, pdist(X))
print(f"Cophenetic Correlation Coefficient: {c:.4f}, {coph_dists}")
