from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models
car_classifier = tf.keras.models.load_model("car_bike_classification_model.h5")
damage_detector = tf.keras.models.load_model("car_damage_model.h5")
location_detector = tf.keras.models.load_model("car_damage_location_model.h5")
severity_classifier = tf.keras.models.load_model("car_damage_severity_model.h5")

location_labels = ["Headlamp/Brake Light", "Front Bumper", "Hood", "Door", "Rear Bumper"]
severity_labels = ['minor', 'moderate', 'severe']
repair_costs = [10000, 50000, 100000]

def prepare_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    temp_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)
    
    try:
        img = prepare_image(temp_path)
        car_prediction = car_classifier.predict(img)[0][0]
        is_car = car_prediction > 0.5

        if not is_car:
            return jsonify({
                "result": "The uploaded image is not a car.",
                "confidence": float(car_prediction)
            })

        damage_prediction = damage_detector.predict(img)[0][0]
        is_damaged = damage_prediction > 0.5

        location_predictions = location_detector.predict(img)[0]
        damage_locations = [
            {"location": location_labels[i], "confidence": float(prob)}
            for i, prob in enumerate(location_predictions) if prob > 0.5
        ]

        severity_predictions = severity_classifier.predict(img)[0]
        severity_index = np.argmax(severity_predictions)
        severity_label = severity_labels[severity_index]
        severity_confidence = float(severity_predictions[severity_index])

        estimated_cost = severity_confidence * repair_costs[severity_index]

        result = {
            "is_car": bool(is_car),
            "is_damaged": bool(is_damaged),
            "damage_confidence": float(damage_prediction),
            "damage_locations": damage_locations,
            "damage_severity": {
                "label": severity_label,
                "confidence": float(severity_confidence)
            },
            "estimated_repair_cost": float(estimated_cost)
        }
    finally:
        os.remove(temp_path)

    return jsonify(result)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
