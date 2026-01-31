from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 
import pandas as pd

#Dataset di esempio 
testi = [ 
    "email promozionale sconto imperdibile",
    "offerta esclusiva clicca qui",
    "appuntamento confermato domani alle 10",
    "riunione con il team progetto ricerca",
    "vincita immediata ritira il premio ora"
]

etichette = ["spam","spam","non_spam","non_spam","spam"]


#Rappresentazione binaria della feature 
#1 - presente nel testo, 0 - non è presente
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(testi)

X_train,X_test,y_train,y_test = train_test_split(X,etichette,test_size=0.4,random_state=42)

#Addestramento modello 
model = BernoulliNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#Valutazione
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Matrice di confusione:\n",confusion_matrix(y_test,y_pred))
print("\nReport di classificazione:\n",classification_report(y_test,y_pred))

print("\nProbabilità predette")

for testo, probs in zip(X_test.toarray(), model.predict_proba(X_test)):
    print(probs)

feature_names = vectorizer.get_feature_names_out()
print(feature_names)
df_feature = pd.DataFrame(model.feature_log_prob_.T,index=feature_names,columns=model.classes_)
print("\nParole più indicative per classe:\n")
print(df_feature.sort_values(by="spam",ascending=False))