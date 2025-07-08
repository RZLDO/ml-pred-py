# app.py
from flask import Flask, request, jsonify
from ml.predict import predict_diagnosis
from ml.train import train_and_save_model

app = Flask(__name__)

@app.route("/ml/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        result = predict_diagnosis(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({ "error": str(e) }), 500

@app.route("/ml/retrain", methods=["GET"])
def retrain(): 
    try:
        result = train_and_save_model() 
        return jsonify(result)
    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(debug=True)
