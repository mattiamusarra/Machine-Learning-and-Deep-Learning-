from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import umap 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='whitegrid')
data = load_digits()
X = data.data
y = data.target 
X_scaled = StandardScaler().fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

reducer  = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='viridis', ax=ax1)
ax1.set_title('Visualizzazione t-SNE')
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=y, palette='magma', ax=ax2)
ax2.set_title("Visualizzazione UMAP")
plt.show()