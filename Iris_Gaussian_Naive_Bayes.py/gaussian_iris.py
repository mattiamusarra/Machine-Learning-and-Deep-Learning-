from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report 




#Caricare il dataset Iris 
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Feature names: {feature_names}")
print(f"Target names: {target_names}")


#Suddividiamo il dataset in training e test set  70 - 30 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#Inizializzazione e addestramento modello 
model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy del modello Gaussian Naive Bayes: {accuracy:.2f}")
print("\nReport di classificazione:")
print(classification_report(y_test,y_pred,target_names=target_names))

#Esempio di predizione con probabilità 
#Prendiamo un campione di test 
sample = X_test[0].reshape(1,-1)
sample_class = y_test[0]

pred_class_idex = model.predict(sample)[0]
pred_class_name = target_names[pred_class_idex]

probabilities = model.predict_proba(sample)
print("Campione (features):",sample)
print("Classe reale:", target_names[sample_class])
print("Classe predetta:", pred_class_name)
print("Probabilità (posteriori)", probabilities[0])
