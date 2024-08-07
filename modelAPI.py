import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import h5py

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello"


@app.route("/predict", methods = ['POST'])
def predict():
    file_path = "model_drive_style.h5"
    try:
        with h5py.File(file_path, 'r') as f:
            # Loading a model
            model = tf.keras.models.load_model(file_path)
            print("HDF5 file is accessible.")
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
    except OSError as e:
        raise OSError(f"Unable to open the file: {e}")


if __name__ == '__main__':
    app.run(debug=True)