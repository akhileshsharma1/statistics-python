import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

df = pd.read_csv("excel_file.csv")

x = sm.add_constant(df[['X1', 'X2', 'X3']])
y = df['Y']

# Fit the regression model
model = sm.OLS(y, x).fit()

# Get residuals
residuals = model.resid

# Plot the normal probability plot
qqplot(residuals, line='s')
plt.title('Normal Probability Plot of Residuals')
plt.show()