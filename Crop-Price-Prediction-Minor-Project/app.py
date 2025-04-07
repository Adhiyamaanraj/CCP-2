from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

try:
    random_forest_model = joblib.load("C:\\Users\\adhiy\\Documents\\ccp_new\\Crop-Price-Prediction-Minor-Project\\random_forest_model.joblib")
except Exception as e:
    print(f"Error loading model: {e}")
    random_forest_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not random_forest_model:
        return "Error: Model could not be loaded."

    crop = request.form.get('crop')
    location = request.form.get('location')
    try:
        rainfall = float(request.form.get('rainfall', 0))
        temperature = float(request.form.get('temperature', 0))
        humidity = int(request.form.get('humidity', 0))
    except ValueError:
        return "Invalid input values. Please enter valid numbers."


    crop_mapping = {'Chilli': 0, 'Groundnut': 1, 'Maize': 2, 'Rice': 3, 'Sugarcane': 4}
    location_mapping = {'Chittoor': 0, 'Guntur': 1, 'Kadapa': 2, 'Nellore': 3, 'Vijayawada': 4}

    crop_encoded = crop_mapping.get(crop, -1)
    location_encoded = location_mapping.get(location, -1)

    if crop_encoded == -1 or location_encoded == -1:
        return "Invalid crop or city selected. Please try again."


    today = pd.Timestamp.today()
    next_12_months = pd.date_range(today, periods=12, freq='MS')


    next_12_months_df = pd.DataFrame({
        'Location': [location_encoded] * 12,
        'Crop': [crop_encoded] * 12,
        'Rainfall': [rainfall] * 12,
        'Temperature': [temperature] * 12,
        'Humidity': [humidity] * 12,
        'Month': next_12_months.month.tolist(),
        'Year': next_12_months.year.tolist()
    })


    predicted_prices_next_12_months = random_forest_model.predict(next_12_months_df)


    price_data = [
        {'month': date.strftime('%B %Y'), 'price': round(price, 2)}
        for date, price in zip(next_12_months, predicted_prices_next_12_months)
    ]

    result = {'crop': crop, 'price_data': price_data}


    crop_info = [
        {'crop': 'Rice', 'info': 'Rice is a key crop in Andhra Pradesh, mainly grown in East Godavari, Krishna, and Guntur. The Kharif season is the primary growing season.'},
        {'crop': 'Groundnut', 'info': 'Groundnut is mainly cultivated in Anantapur and Kurnool, thriving in semi-arid regions.'},
        {'crop': 'Sugarcane', 'info': 'Sugarcane is a major cash crop grown in Krishna and Guntur, supporting sugar and ethanol industries.'},
        {'crop': 'Maize', 'info': 'Maize is grown in Anantapur and Prakasam, serving as a major food and fodder crop.'},
        {'crop': 'Chilli', 'info': 'Chilli cultivation is centered in Guntur, known for producing high-quality mirchi varieties.'}
    ]

    return render_template('result.html', result=result, crop_info=crop_info)

if __name__ == '__main__':
    app.run(debug=True)
