from matplotlib.image import imread
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

img_raw = imread('immagine.jpg')
img_gray = img_raw.mean(axis=2) #La trasformiamo in un immagine in grigio

n_comp = 50 
pca = PCA(n_components=n_comp)
img_transformed = pca.fit_transform(img_gray)
img_reconstructed = pca.inverse_transform(img_transformed)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img_gray, cmap='gray'); plt.title('Originale')
plt.subplot(1,2,2)
plt.imshow(img_reconstructed,cmap='gray')
plt.title(f'Compressa con {n_comp} componenti')
plt.show()