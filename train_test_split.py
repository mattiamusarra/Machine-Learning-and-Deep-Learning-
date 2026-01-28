from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


data = load_iris()
x = data.data
y = data.target


#Suddividiamo in Train (70%), temp(30%)
x_train,x_temp,y_train,y_temp = train_test_split(
    x,y,test_size=0.3,random_state=42,stratify=y
)

#Poi suddividiamo il restante 30% in Validation (20%) e Test(10%)
#Proporzione relativa: Validation = 2/3 di temp, Test = 1/3 di temp
x_valid,x_test, y_valid, y_test = train_test_split(
    x_temp,y_temp,test_size=0.33,random_state=42,stratify=y_temp
    )
print(f"All data: {len(x)}")
print(f"Training set: {len(x_train)} campioni")
print(f"Validation set: {len(x_valid)} campioni")
print(f"Test set: {len(x_test)} campioni")