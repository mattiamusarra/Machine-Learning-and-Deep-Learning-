import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt 

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Standardizzare le feature 
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_range = range(1,26)
k_scores = []

print("Ricerca del k ottimale tramite Cross-Validation:")
for k in k_range:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv,X_train_scaled,y_train,cv=10,scoring="accuracy")
    k_scores.append(scores.mean())


optimal_k = k_range[np.argmax(k_scores)]
print(f"Il valore ottimale di k è: {optimal_k}")

plt.figure(figsize=(10,6))
plt.plot(k_range,k_scores,color="blue",linestyle='dashed',marker='o',markerfacecolor='red',markersize=8)
plt.title("Accuratezza vs. Valore di K(Metodo Elbow)")
plt.xlabel("Valore di K")
plt.ylabel("Accuretezza medai (Cross-Validation)")
plt.grid(True)

plt.vlines(optimal_k,ymin=min(k_scores),ymax=max(k_scores),linestyles='dotted',colors='red')
plt.show()
#Addestriamo il modello con il k ottimale 
model = KNeighborsClassifier(n_neighbors=optimal_k, weights ='distance', metric ='euclidean')
model.fit(X_train_scaled, y_train) 
y_pred = model.predict(X_test_scaled)

#Valutazione 
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy sul  Test Set (k={optimal_k}, weighted): {accuracy:.4f}")
print(classification_report(y_test,y_pred,target_names=iris.target_names))
