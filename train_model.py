import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv('housing.csv')

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing values for categorical columns with mode
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separate features and target variable
X = df.drop('price', axis=1)  # Assuming 'price' is the target variable
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and feature names for consistency
joblib.dump(model, 'housing_price_model.pkl')
joblib.dump(X_train.columns, 'model_features.pkl')  # Save feature columns

print("Model training completed and saved.")
