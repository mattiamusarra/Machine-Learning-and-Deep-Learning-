import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import make_circles, make_blobs 
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering, OPTICS

#Creazinone dei dati : cerchi + blob denso 
X1, _ = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[1.5,1.5]], cluster_std=0.2, random_state=42)
X = np.vstack([X1,X2])
X = StandardScaler().fit_transform(X) #Fondamentale per spectral e GMM

fig, axes = plt.subplots(2,2,figsize=(12,10))

#1) GMM 
gmm = GaussianMixture(n_components=3).fit_predict(X)
axes[0,0].scatter(X[:,0], X[:,1], c=gmm, cmap='viridis', s=20)
axes[0,0].set_title("Gaussian Mixture (GMM)")

#2) Spectral  
spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42).fit_predict(X)
axes[0,1].scatter(X[:,0], X[:,1], c=spectral, cmap='tab10', s=20)
axes[0,1].set_title("Spectral Clustering")

#3) OPTICS (DENSITA VARIABILE) 
optics = OPTICS(min_samples=10).fit_predict(X)
axes[1,0].scatter(X[:,0], X[:,1], c=optics, cmap='plasma', s=20)
axes[1,0].set_title("OPTICS (Basato su Densità)")

#4) Mean Shift
ms = MeanShift(bandwidth=0.8).fit_predict(X)
axes[1,1].scatter(X[:,0], X[:,1], c=ms, cmap='coolwarm', s=20)
axes[1,1].set_title("Mean Shift")

plt.tight_layout()
plt.show()