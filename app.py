from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Model ve encoder'ları yükle
try:
    model = joblib.load('model/traffic_model.pkl')
    le_day = joblib.load('model/le_day.pkl')
    le_weather = joblib.load('model/le_weather.pkl')  # le_week yerine le_weather
    le_traffic = joblib.load('model/le_traffic.pkl')
except FileNotFoundError as e:
    print(f"Hata: Model dosyası bulunamadı: {e}")
    exit()

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Formdan verileri al
        day = request.form['day']
        hour = int(request.form['hour'])
        weather = request.form['weather']
        temp = int(request.form['temperature'])

        # Veriyi model için hazırla
        day_encoded = le_day.transform([day])[0]
        weather_encoded = le_weather.transform([weather])[0]
        input_data = pd.DataFrame({
            'Gün': [day_encoded],
            'Saat': [hour],
            'Hava Durumu': [weather_encoded],
            'Sıcaklık': [temp]
        })

        # Tahmin yap
        prediction = model.predict(input_data)
        prediction_label = le_traffic.inverse_transform(prediction)[0]

        return render_template('index.html', prediction=f"Tahmini Trafik Durumu: {prediction_label}")
    except Exception as e:
        return render_template('index.html', prediction=f"Hata: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)