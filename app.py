import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import tensorflow as tf
import os
import sys
import gc

# Initialize App
app = Flask(__name__)

def log(msg):
    print(msg, file=sys.stdout)
    sys.stdout.flush()

log("STARTING APP SYSTEM...")

try:
    log("Loading Keras Model...")
    
    # Robust check for filename variations
    if os.path.exists('accident_model.keras'):
        model_path = 'accident_model.keras'
    elif os.path.exists('Accident Model.keras'):
        model_path = 'Accident Model.keras'
    else:
        raise FileNotFoundError("Could not find 'accident_model.keras' or 'Accident Model.keras'")
        
    model = tf.keras.models.load_model(model_path)
    log(f"Model loaded from: {model_path}")

    log("Loading Preprocessor...")
    if not os.path.exists('preprocessor.pkl'):
        raise FileNotFoundError("Could not find 'preprocessor.pkl'")
        
    preprocessor = joblib.load('preprocessor.pkl')
    log("Preprocessor loaded.")

except Exception as e:
    log(f"CRITICAL LOAD ERROR: {e}")

# Show the Website
@app.route('/')
def home():
    return render_template('index.html')

# Handle the Form
@app.route('/predict', methods=['POST'])
def predict():
    gc.collect()
    log("📥 REQUEST RECEIVED: Starting prediction process...")
    try:
        # Get data
        input_data = {
            'speed_limit': float(request.form['speed_limit']),
            'num_lanes': int(request.form['num_lanes']),
            'curvature': float(request.form['curvature']),
            'num_reported_accidents': int(request.form['num_reported_accidents']),
            'weather': request.form['weather'],
            'lighting': request.form['lighting'],
            'road_type': request.form['road_type'],
            'time_of_day': request.form['time_of_day'],

            'road_signs_present': int(request.form.get('road_signs_present', 0)),
            'public_road': int(request.form.get('public_road', 0)),
            'holiday': int(request.form.get('holiday', 0)),
            'school_season': int(request.form.get('school_season', 0))
        }
        log("Data extracted from form.")

        # DataFrame
        features_df = pd.DataFrame([input_data])
        log("DataFrame created.")

        # Preprocess
        log("Scaling data (using preprocessor)...")
        if 'preprocessor' not in globals():
            raise Exception("Preprocessor was not loaded properly at startup.")
            
        processed_features = preprocessor.transform(features_df)
        log("Data scaled successfully.")

        # Predict
        log("Sending to Neural Network...")
        if 'model' not in globals():
            raise Exception("Model was not loaded properly at startup.")

        prediction = model.predict(processed_features)
        risk_score = prediction[0][0]
        log(f"Prediction Result: {risk_score}")


        percentage = "{:.2f}".format(risk_score * 100)
        
        # Determine Status
        if risk_score > 0.7:
            result_text = f"DANGER: High Accident Risk ({percentage}%)"
        elif risk_score > 0.4:
            result_text = f"CAUTION: Moderate Risk ({percentage}%)"
        else:
            result_text = f"SAFE: Low Accident Risk ({percentage}%)"

        log(f"Returning: {result_text}")
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        log(f"ERROR DURING PREDICTION: {e}")
        return render_template('index.html', prediction_text=f"System Error: {str(e)}")

if __name__ == "__main__":

    app.run(debug=True, port=7860)