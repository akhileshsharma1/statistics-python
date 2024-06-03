import pandas as pd 
from sklearn.linear_model import LinearRegression


df = pd.read_csv("excel_file.csv")


x = df[['X1', 'X2', 'X3']]
y = df['Y']

model = LinearRegression().fit(x, y)

new_data = pd.DataFrame({'X1': [8], 'X2': [14], 'X3': [25]})

predicted_y = model.predict(new_data)

print(f'Estimated Y for X1=8, X2=14, X3=25 is: {predicted_y[0]:.4f}')