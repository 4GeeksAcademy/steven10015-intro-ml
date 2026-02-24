from flask import Flask, request, jsonify
import pickle          
import numpy as np
import os
app = Flask(__name__)
# -----------------------------
# 1. Cargar modelo
# -----------------------------
model_path = '/workspaces/steven10015-intro-ml/models/iris_model.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# -----------------------------
# 2. Ruta de prueba
# -----------------------------
@app.route('/')
def home():
    return "¡Flask funcionando con Iris Model!"

# -----------------------------
# 3. Ruta de predicción
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convertir entrada a array 2D
    features = np.array([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]])

    # Hacer predicción
    prediction = model.predict(features)

    return jsonify({'prediction': prediction[0]})

# -----------------------------
# 4. Ejecutar app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)