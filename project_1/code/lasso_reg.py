import pandas as pd

file_path = '/Users/rahulramakrishnan/Documents/spring_2024/STAT_4214/code/Breast_Cancer.csv'
data = pd.read_csv(file_path)

data_encoded = pd.get_dummies(data, drop_first=True)  # drop_first to avoid dummy variable trap

X = data_encoded.drop('Survival Months', axis=1)  # features
y = data_encoded['Survival Months']  # target

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

coefficients = regressor.coef_
intercept = regressor.intercept_

coeff_df = pd.DataFrame(coefficients, X.columns, columns=['Coefficient'])

print(coeff_df)

from sklearn.linear_model import Ridge, Lasso

ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X_train, y_train)
ridge_y_pred = ridge_reg.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)

lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(X_train, y_train)
lasso_y_pred = lasso_reg.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)

print(f'Ridge Regression Mean Squared Error: {ridge_mse}')
print(f'Ridge Regression R^2 Score: {ridge_r2}')
print(f'Lasso Regression Mean Squared Error: {lasso_mse}')
print(f'Lasso Regression R^2 Score: {lasso_r2}')

lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(X_train, y_train)
lasso_y_pred = lasso_reg.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)

print(f'Ridge Regression Mean Squared Error: {ridge_mse}')
print(f'Ridge Regression R^2 Score: {ridge_r2}')
print(f'Lasso Regression Mean Squared Error: {lasso_mse}')
print(f'Lasso Regression R^2 Score: {lasso_r2}')

lasso_coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": lasso_reg.coef_})
important_features = lasso_coefficients[lasso_coefficients["Coefficient"] != 0]
print(important_features.sort_values(by="Coefficient", ascending=False))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.scatter(y_test, lasso_y_pred, color='red', label='Lasso Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Lasso Regression')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=important_features.sort_values(by="Coefficient", ascending=False))
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Important Features for Lasso Regression')
plt.show()
