import numpy as np
from joblib import load

model = load("ml/model.pkl")
scaler = load("ml/scaler.pkl")

def predict_diagnosis(input_data: dict):
    try:
        features = np.array([
            input_data["radiusMean"],
            input_data["textureMean"],
            input_data["perimeterMean"],
            input_data["areaMean"],
            input_data["smoothnessMean"],
            input_data["compactnessMean"],
            input_data["concavityMean"]
        ]).reshape(1, -1)

        features_scaled = scaler.transform(features)  

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        diagnosis = "M" if prediction == 1 else "B"
        probability = round(probabilities[prediction], 4)

        return {
            "error": False,
            "message": "Prediction success",
            "data": {
                "diagnosis": diagnosis,
                "probability": probability,  
                "raw_probabilities": {
                    "B": round(probabilities[0], 4),
                    "M": round(probabilities[1], 4)
                }
            }
        }

    except Exception as e:
        return {
            "error": True,
            "message": "Prediction failed",
            "data": {
                "error_message": str(e)
            }
        }
