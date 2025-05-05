from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'feature{i}']) for i in range(1, 5)]
        prediction = model.predict([features])[0]
        return render_template('index.html', prediction_text=f'Hasil Prediksi: {prediction}')
    except:
        return render_template('index.html', prediction_text="Input tidak valid.")

if __name__ == '__main__':
    app.run(debug=True)
