from sklearn.model_selection import KFold, cross_val_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_iris




#Dataset di esempio 
x,y = load_iris(return_X_y=True)

#Definizione della F-Fold (con 5 partizioni) 
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Modello da valutare 
model = LogisticRegression(max_iter=200)

#Cross validation
scores = cross_val_score(model,x,y,cv=kf)

print("Accuracy per ogni fold:", scores)
print("Accuracy media:", scores.mean())