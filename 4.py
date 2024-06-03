import pandas as pd 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv("excel_file.csv")

x = df[['X1', 'X2', 'X3']]
y = df['Y']

model = sm.OLS(y, sm.add_constant(x)).fit()

standard_errors = model.bse

print("Standard Error of B0 (Intercept):", standard_errors['const'])
print("Standard Error of B1 (X1):", standard_errors['X1'])
print("Standard Error of B2 (X2):", standard_errors['X2'])
print("Standard Error of B3 (X3):", standard_errors['X3'])

residual_standard_error = (model.resid**2).mean()**0.5
print("Residual Standard Error of the Estimate:", residual_standard_error)
