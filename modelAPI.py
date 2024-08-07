import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Loading a model
model = tf.keras.models.load_model("model_drive_style.h5")

@app.route('/')
def home():
    return "Hello"


@app.route("/predict", methods = ['POST'])
def predict():
    data = request.get_json(force=True)
    print("Received data:", data)
    features = np.array([[
        entry["AccX"],
        entry["AccY"],
        entry["AccZ"],
        entry["GyroX"],
        entry["GyroY"],
        entry["GyroZ"]
    ] for entry in data])

    prediction = model.predict(features)

    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run(debug=True)