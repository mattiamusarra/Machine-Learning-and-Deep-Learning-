from sklearn.datasets import load_breast_cancer
from sklearn.model_selection  import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


#Caricamrento del dataset 
data = load_breast_cancer()
X = data.data
y= data.target

#Suddivisione del dataset in training e test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

#Normalizziamo le feature 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma="scale")
svm_rbf.fit(X_train, y_train)

y_pred = svm_rbf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred),3))
print("\nClassification Report:\n", classification_report(y_test, y_pred,target_names=data.target_names))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - SVM with RBF Kernel")
plt.show()
