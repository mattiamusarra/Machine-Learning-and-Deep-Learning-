from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import load_iris 
import numpy as np


#Dataset di esempio 
X,y = load_iris(return_X_y=True)


#Modello basato su alberi decisionali 
model = RandomForestClassifier()

#Spazio di ricerca degli iperparametri 
param_dist = {
    'n_estimators': np.arange(50,300,50),
    'max_depth': [None, 5,10,20],
    'min_samples_split': np.arange(2,10)
}
#Random search con 20 combinazioni casuali
random_search = RandomizedSearchCV(
    model, param_distributions=param_dist,
    n_iter=20, cv=5, random_state=42,n_jobs=-1
)

random_search.fit(X,y)
print("Migliori parametri trovati:", random_search.best_params_)
print("Accuracy media CV:", random_search.best_score_)