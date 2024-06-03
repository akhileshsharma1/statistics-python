import pandas as pd 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv("excel_file.csv")

x = sm.add_constant(df[['X1', 'X2', 'X3']])
y = df['Y']

# Fit the regression model
model = sm.OLS(y, x).fit()

# Print the summary
print(model.summary())

