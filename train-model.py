import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv('car data.csv')

# Clean string columns
for col in ['Fuel_Type', 'Seller_Type', 'Transmission', 'Car_Name']:
    df[col] = df[col].astype(str).str.strip().fillna('Unknown')

df['Selling_Price'] = df['Selling_Price'] * 100000

df['Age'] = 2025 - df['Year'] 
df['Kms_Driven'] = df['Kms_Driven'].clip(upper=300000) 

X = df[['Age', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Car_Name']]
y = df['Selling_Price']

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('fuel', OneHotEncoder(handle_unknown='ignore'), ['Fuel_Type']),
    ('seller', OneHotEncoder(handle_unknown='ignore'), ['Seller_Type']),
    ('trans', OneHotEncoder(handle_unknown='ignore'), ['Transmission']),
    ('car', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['Car_Name']),
    ('scale', StandardScaler(), ['Age', 'Kms_Driven'])
])

model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', HistGradientBoostingRegressor(max_depth=7, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, color='royalblue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal (y = x)')
plt.xlabel('Actual Selling Price (₹)')
plt.ylabel('Predicted Selling Price (₹)')
plt.title(f'Predicted vs Actual Prices\n$R^2$ = {r2:.2f}, RMSE = ₹{int(rmse)}')
plt.legend()
plt.tight_layout()
plt.show()

# Save model and column order
joblib.dump(model_pipeline, 'model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')

