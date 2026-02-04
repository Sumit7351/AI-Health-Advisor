from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import json
import os
import traceback
import model  # Import the model module to use its predict function if needed, or just load artifacts directly

app = Flask(__name__)
CORS(app)

# Load artifacts
print("Loading model and data...")
try:
    model_obj, le, explainer = model.load_artifacts()
    
    with open("disease_info.json", "r") as f:
        disease_info = json.load(f)
        
    # Get symptom list from dataset columns
    df = pd.read_csv("dataset.csv")
    all_symptoms = df.drop("Disease", axis=1).columns.tolist()
    print("Model and data loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model_obj = None

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": all_symptoms})

@app.route('/predict', methods=['POST'])
def predict():
    if not model_obj:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    selected_symptoms = data.get('symptoms', [])
    
    if not selected_symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
        
    # Create input dictionary
    input_dict = {sym: 0 for sym in all_symptoms}
    for sym in selected_symptoms:
        if sym in input_dict:
            input_dict[sym] = 1
            
    # Use model.predict_disease logic
    try:
        result = model.predict_disease(input_dict)
        
        # Add disease info
        disease_name = result["disease"]
        info = disease_info.get(disease_name, {})
        
        response = {
            "prediction": result,
            "info": info
        }
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
