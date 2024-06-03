import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("excel_file.csv")

x = df[['X1', 'X2', 'X3']]
y = df['Y']

model = LinearRegression()

# Fit the model
model.fit(x, y)

# Get the coefficients (slope) and intercept
coefficients = model.coef_
intercept = model.intercept_

# Make predictions for the scatter plot
predictions = model.predict(x)

# Calculate R^2 value
r2_value = r2_score(y, predictions)

# Scatter plot
plt.scatter(y, predictions)
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red', label='Perfect Fit')
plt.title('Scatter Plot with Regression Line')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')

# Display regression equation on the plot
equation_text = f'Regression Equation:\nY = {intercept:.2f} + {coefficients[0]:.2f}*X1 + {coefficients[1]:.2f}*X2 + {coefficients[2]:.2f}*X3'
plt.text(min(y), max(predictions), equation_text, fontsize=10, verticalalignment='bottom', horizontalalignment='left')

# Display R^2 value on the plot
r2_text = f'R^2 Value: {r2_value:.2f}'
plt.text(min(y), max(predictions)-2, r2_text, fontsize=10, verticalalignment='bottom', horizontalalignment='left')

plt.show()

for col in ['X1', 'X2', 'X3']:
    x = sm.add_constant(df[col])
    y = df['Y']
    model = sm.OLS(y, x).fit()
    plt.scatter(df[col], df['Y'], label='Actual data')
    plt.plot(df[col], model.predict(x), color='red', label='Regression line')
    plt.xlabel(col)
    plt.ylabel('Y')
    plt.title(f'Scatter Diagram and Regression Line for {col}')
    plt.legend()
    plt.show()