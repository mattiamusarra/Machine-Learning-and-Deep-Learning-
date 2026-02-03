import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import matplotlib.pyplot as plt

data = {
    'TV' : [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio' : [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper' : [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales' : [22.1, 10.4, 9.3, 18.5, 12.9]
}

df = pd.DataFrame(data)

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

print("Intercept : ", model.intercept_)
print("Coefficients:", model.coef_)

mse = mean_squared_error(y,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y,y_pred)

print("RMSE :",rmse)
print("R2 :",r2)

X_vif = sm.add_constant(X)
vif = pd.DataFrame()
vif['variable'] = X_vif.columns
vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif)

plt.figure(figsize=(7,5))
plt.scatter(y,y_pred)
plt.plot([y.min(), y.max()],[y.min(),y.max()],linestyle='--')
plt.xlabel("Valori osservati")
plt.ylabel("Valori predetti")
plt.title("Valori osservati vs valori predetti")
plt.grid(True)
plt.show()

residuals = y - y_pred
plt.figure(figsize=(7,5))
plt.scatter(y_pred,residuals)
plt.axhline(0, linewidth=2)
plt.xlabel("Valori predetti")
plt.ylabel("Residui")
plt.title("Grafico dei residui")
plt.grid(True)
plt.show()

coefficients = model.coef_
features = X.columns 
plt.figure(figsize=(7,5))
plt.bar(features, coefficients)
plt.axhline(0, linewidth=1)
plt.xlabel("Predittori")
plt.ylabel("Coefficienti")
plt.title("Coefficienti di regressione")
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(7,5))
plt.hist(residuals, bins=10)
plt.xlabel("Residui")
plt.ylabel("Frequenza")
plt.title("Distribuzione dei Residui")
plt.grid(True)
plt.show()