import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'TV' : [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2],
    'Sales' : [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 17.1]
}

df = pd.DataFrame(data)

X = df[['TV']]
y = df['Sales']

print(y)

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)
print("Intercetta :",model.intercept_)
print("Coefficiente :", model.coef_[0])

mse = mean_squared_error(y,y_pred)
r2 = r2_score(y,y_pred)
print("Mean Squared Error: ",mse)
print("R^2 Score: ",r2)

plt.scatter(X,y, color='blue', label='Dati reali')
plt.plot(X,y_pred, color='red', linewidth=2, label='Retta di regressione')
plt.title('Regressione lineare semplice')
plt.xlabel('Investimento in TV')
plt.ylabel('Vendite')
plt.legend()
plt.show()