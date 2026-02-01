from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 


#Creiamo un set di dati di esempio
X,y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modello_svm = SVC(kernel="rbf", C=1.0, gamma="auto")

#addestramento 
modello_svm.fit(X_train_scaled, y_train)
y_pred = modello_svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello SVM: {accuracy},{accuracy*100:.2f}%")
print(modello_svm.support_vectors_)


