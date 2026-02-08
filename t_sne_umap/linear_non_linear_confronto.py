from sklearn.datasets import load_digits 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt 
import seaborn as sns 
import time 


digits = load_digits()
X,y = digits.data, digits.target 
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


start = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
time_tsne = time.time() - start 

start = time.time()
reducer = umap.UMAP(n_neighbors = 15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
time_umap = time.time() - start

fig, axes = plt.subplots(1,3, figsize=(20,6))

titles = ['PCA (lineare)', f"t-SNE (Tempo : {time_tsne:.2f}s)", f"UMAP(Tempo: {time_umap:.2f} s)"]
data_list = [X_pca, X_tsne, X_umap]

for i, ax in enumerate(axes):
    scatter = ax.scatter(data_list[i][:,0], data_list[i][:,1], c=y, cmap='tab10', s=10, alpha=0.7)
    ax.set_title(titles[i])
    if i==2:
        fig.colorbar(scatter, ax=ax, label='Cifra (0-9)')

plt.tight_layout()
plt.show()