import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import tensorflow as tf

# Initialize App
app = Flask(__name__)

# Load the AI Brain (Model) and Translator (Preprocessor)

try:
    model = tf.keras.models.load_model('Accident Model.keras')
    preprocessor = joblib.load('preprocessor.pkl')
    print("✅ Model and Preprocessor Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("Make sure 'accident_model.keras' and 'preprocessor.pkl' are in the folder.")

# Show the Website
@app.route('/')
def home():
    return render_template('index.html')

# Handle the Form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        
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

       
        features_df = pd.DataFrame([input_data])

       
        processed_features = preprocessor.transform(features_df)

        
        prediction = model.predict(processed_features)
        risk_score = prediction[0][0]  

       
        percentage = round(risk_score * 100, 2)
        
        if risk_score > 0.7:
            result_text = f"⚠️ DANGER: High Accident Risk ({percentage}%)"
        elif risk_score > 0.4:
            result_text = f"⚠️ CAUTION: Moderate Risk ({percentage}%)"
        else:
            result_text = f"✅ SAFE: Low Accident Risk ({percentage}%)"

       
        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True, port=5001)