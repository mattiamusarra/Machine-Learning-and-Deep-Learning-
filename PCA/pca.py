import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

#Configurazione 
sns.set_theme(style='whitegrid')

data = load_iris()
X = data.data
y = data.target 

X_Scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2) #Vogliamo passare da 4D a 2D
X_pca = pca.fit_transform(X_Scaled)

df_pca = pd.DataFrame(data=X_pca, columns=['PC1','PC2'])
df_pca['Target'] = data.target_names[y]
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Target', palette='viridis', s=70)
plt.title(f'PCA di Iris: Varianza Totale Spiegata {np.sum(pca.explained_variance_ratio_):.2f}')
plt.show()

print(f"Varianza PC1: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Varianza PC2: {pca.explained_variance_ratio_[1]:.2f}")

pca_full = PCA().fit(X_Scaled)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Numero di Componenti')
plt.ylabel('Varianza Spiegata Cumulativa')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.title('Scree Plot per la selezione delel componenti')
plt.show()
