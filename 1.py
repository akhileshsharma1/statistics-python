import pandas as pd 
from sklearn.linear_model import LinearRegression


df = pd.read_csv("excel_file.csv")


x = df[['X1', 'X2', 'X3']]
y = df['Y']

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print("R-squared:", r_sq)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

y_pred = model.predict(x)
print(f"Predicted response: {y_pred}")
coefficients = model.coef_
intercept = model.intercept_

equation = f"Regression Equation: Y = {model.intercept_:.4f} "
for i, coef in enumerate(model.coef_):
    equation += f"+ ({coef:.4f} * X{i+1})"
print(equation)
