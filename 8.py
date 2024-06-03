import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

df = pd.read_csv("excel_file.csv")

x = sm.add_constant(df[['X1', 'X2', 'X3']])
y = df['Y']

x = sm.add_constant(df[['X1', 'X2', 'X3']])

vif_data = pd.DataFrame()
vif_data["Variable"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

print(vif_data)