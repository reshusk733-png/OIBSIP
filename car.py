import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load the dataset 
# Replace 'car_data.csv' with the actual path to your downloaded file
try:
    df = pd.read_csv('car_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Please ensure the dataset file is in the same directory.")
    # Creating a dummy dataframe for demonstration if file is missing
    data = {
        'Year': [2014, 2013, 2017, 2011, 2014],
        'Present_Price': [5.59, 9.54, 9.85, 4.15, 6.87],
        'Kms_Driven': [27000, 43000, 6900, 5200, 42450],
        'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel'],
        'Seller_Type': ['Dealer', 'Dealer', 'Dealer', 'Individual', 'Dealer'],
        'Transmission': ['Manual', 'Manual', 'Manual', 'Manual', 'Manual'],
        'Selling_Price': [3.35, 4.75, 7.25, 2.85, 4.60]
    }
    df = pd.DataFrame(data)

# 2. Basic Data Exploration
print(df.head())

# 3. Feature Engineering: Calculating the Age of the car
current_year = 2024
df['Car_Age'] = current_year - df['Year']
df.drop('Year', axis=1, inplace=True)

# 4. Splitting Data into Features (X) and Target (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Preprocessing & Model Pipeline
# Identifying categorical columns for OneHotEncoding
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
numerical_features = ['Present_Price', 'Kms_Driven', 'Car_Age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

# Create the pipeline with a Random Forest Regressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6. Training
model_pipeline.fit(X_train, y_train)

# 7. Evaluation
y_pred = model_pipeline.predict(X_test)

print(f"\nModel Performance Metrics:")
print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")

# 8. Visualizing Results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.show()