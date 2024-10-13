from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model and feature names
model = joblib.load('housing_price_model.pkl')
model_features = joblib.load('model_features.pkl')

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = request.form.to_dict()

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess numeric columns (convert to appropriate types)
    numeric_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Handle missing categorical columns and encode them
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Ensure input DataFrame has the same columns as the model
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Make prediction using the loaded model
    prediction = model.predict(input_df)[0]

    return render_template('result.html', prediction=round(prediction, 2))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
