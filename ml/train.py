from joblib import dump
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from db.mysql_connector import load_training_data
import numpy as np
import traceback

def train_and_save_model(model_path="ml/model.pkl", scaler_path="ml/scaler.pkl"):
    try:
        df = load_training_data()
        X = df.drop("diagnosis", axis=1)
        y = LabelEncoder().fit_transform(df["diagnosis"])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dump(scaler, scaler_path)  

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        imbalance_ratio = min(counts) / max(counts)

        print("📊 Class distribution before SMOTE:", class_distribution)

        if imbalance_ratio < 0.7:
            print("⚠️  Detected imbalance, applying SMOTE...")
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        # Train model
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16, 8),
            max_iter=1000,
            early_stopping=True,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Save model
        dump(model, model_path)

        # Evaluate model
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }

        # Mapping class
        mapped_distribution = {
            "Melignant": class_distribution.get(0, 0),
            "Benigna": class_distribution.get(1, 0)
        }

        return {
            "error": False,
            "message": "Model trained successfully",
            "data": {
                "metrics": metrics,
                "class_distribution": mapped_distribution,
            }
        }

    except Exception as e:
        print("❌ Training failed:")
        print(f"Error: {e}")
        traceback.print_exc()

        return {
            "error": True,
            "message": "Training failed",
            "data": {
                "error_message": str(e)
            }
        }

