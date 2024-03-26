import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

file_path = '/Users/rahulramakrishnan/Documents/spring_2024/STAT_4214/code/Breast_Cancer.csv'
data = pd.read_csv(file_path)

data_encoded = pd.get_dummies(data, drop_first=True)
significant_features = ['T Stage _T3', 'T Stage _T4', 'Estrogen Status_Positive', 'Age', 'Tumor Size']  # Example: replace with your actual significant features

X_significant = data_encoded[significant_features]
y = data_encoded['Survival Months']

scaler = StandardScaler()
X_significant_scaled = scaler.fit_transform(X_significant)

X_train_sig, X_test_sig, y_train, y_test = train_test_split(X_significant_scaled, y, test_size=0.2, random_state=42)

model_sig = LinearRegression()
model_sig.fit(X_train_sig, y_train)

# Predict on the test set
y_pred_sig = model_sig.predict(X_test_sig)

mse_sig = mean_squared_error(y_test, y_pred_sig)
r2_sig = r2_score(y_test, y_pred_sig)

print(f'Simplified Model Mean Squared Error (MSE): {mse_sig}')
print(f'Simplified Model R-squared (R²): {r2_sig}')

coefficients = pd.DataFrame(model_sig.coef_, significant_features, columns=['Coefficient'])
print(coefficients)
