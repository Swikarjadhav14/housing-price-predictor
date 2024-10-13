from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('housing_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    input_df = pd.DataFrame([input_data])
    input_df.fillna(input_df.median(), inplace=True)
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_df)
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
