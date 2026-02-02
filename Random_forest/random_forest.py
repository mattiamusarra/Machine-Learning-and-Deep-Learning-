import pandas as pd
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 

iris = load_iris()
X= iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

clf = RandomForestClassifier(n_estimators=100,random_state=42)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#valutazione 
accuracy = accuracy_score(y_test,y_pred)
print(f"Acuracy : {accuracy*100:.2f}%")
print(f"\nClassification report : {classification_report(y_test,y_pred, target_names=iris.target_names)}")
print(f"\nConfusion Matrix :\n {confusion_matrix(y_test,y_pred)}")

feature_imp = pd.Series(clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
print(feature_imp)

plt.figure(figsize=(12,8))
plot_tree(clf.estimators_[0], feature_names=iris.feature_names, class_names= iris.target_names, filled=True, rounded=True)
plt.title("Esempio di un albero della Random Forest")
plt.show()