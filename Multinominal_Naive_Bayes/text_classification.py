from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 


#Dataset di esempio 
testi = [ 
    "email promozionale sconto imperdibile",
    "offerta esclusiva clicca qui",
    "appuntamento confermato domani alle 10",
    "riunione con il team progetto ricerca",
    "vincita immediata ritira il premio ora"
]

etichette = ["spam","spam","non_spam","non_spam","spam"]


#Vettorizzazione del testo
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(testi)
y = etichette
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)

model = MultinomialNB()
model.fit(X_train,y_train)

#Predizione
y_pred = model.predict(X_test)

#Valutazione
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Matrice di confusione:\n",confusion_matrix(y_test,y_pred))
print("\nReport di classificazione:\n",classification_report(y_test,y_pred))

