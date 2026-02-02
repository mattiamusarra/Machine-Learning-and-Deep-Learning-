from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from matplotlib import pyplot as plt 
import numpy as np 
#Dataset di esempio 
X,y = load_iris(return_X_y=True)
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

#Modello Decision Tree con criterio Gini 
clf = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=42)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
#plt.figure(figsize=(10,6))
#plot_tree(clf,filled=True, feature_names=load_iris().feature_names,class_names=load_iris().target_names)
#plt.show()

#Valutiamo le performance 
print("Accuracy", accuracy_score(y_test,y_pred))
print("\n Classification REport:\n",classification_report(y_test,y_pred,target_names=load_iris().target_names))
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=load_iris().target_names)
disp.plot(cmap="Blues")
plt.title("Matrice di confusione - Decision Tree")
plt.show()

print(np.round(clf.feature_importances_,3))