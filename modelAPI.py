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
        result = prediction.tolist()
        type_1 = 'AGGRESSIVE'
        type_2 = 'NORMAL'
        type_1_per = [i for i in result if i == type_1] 
        type_2_per = [i for i in result if i == type_2] 
        res = [{
            'AGGRESSIVE' : (len(type_1_per) / len(result)) * 100,
            'NORMAL' : (len(type_2_per) / len(result)) * 100
        }]
        return jsonify(res)
    except OSError as e:
        raise OSError(f"Unable to open the file: {e}")


if __name__ == '__main__':
    app.run(debug=True)