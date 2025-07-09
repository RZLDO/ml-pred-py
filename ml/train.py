from joblib import dump
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from db.mysql_connector import load_training_data
import numpy as np
import traceback

def train_and_save_model(model_path="ml/model.pkl", scaler_path="ml/scaler.pkl"):
    try:
        df = load_training_data()
        
        # Feature selection/engineering bisa ditambahkan di sini
        X = df.drop("diagnosis", axis=1)
        y = LabelEncoder().fit_transform(df["diagnosis"])
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dump(scaler, scaler_path)
        
        # Train-test split dengan stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        imbalance_ratio = min(counts) / max(counts)
        
        print("📊 Class distribution before SMOTE:", class_distribution)
        print(f"📊 Imbalance ratio: {imbalance_ratio:.3f}")
        
        # Apply SMOTE if needed
        if imbalance_ratio < 0.7:
            print("⚠️ Detected imbalance, applying SMOTE...")
            sm = SMOTE(random_state=42, k_neighbors=3)  # Reduced k_neighbors for small datasets
            X_train, y_train = sm.fit_resample(X_train, y_train)
            
            # Print new distribution
            unique_new, counts_new = np.unique(y_train, return_counts=True)
            new_distribution = {int(k): int(v) for k, v in zip(unique_new, counts_new)}
            print("📊 Class distribution after SMOTE:", new_distribution)
        
        # Grid search for hyperparameter tuning
        print("🔍 Starting hyperparameter tuning...")
        
        # Optimized parameter grid (reduced for faster training)
        param_grid = {
            'hidden_layer_sizes': [
                (100, 50),
                (150, 100, 50),
                (200, 100)
            ],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0005, 0.001],
            'learning_rate_init': [0.0005, 0.001],
            'solver': ['adam']  # Remove lbfgs as it's slower for large datasets
        }
        
        # Base model for grid search (faster settings)
        base_model = MLPClassifier(
            max_iter=1500,  # Reduced from 3000
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,  # Reduced from 20
            tol=1e-4,
            random_state=42
        )
        
        # Grid search with cross-validation (faster settings)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,  # Reduced from 5 folds
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        print(f"🎯 Best parameters: {grid_search.best_params_}")
        print(f"🎯 Best CV score: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters
        print("🚀 Training final model...")
        best_model.fit(X_train, y_train)
        
        # Save model
        dump(best_model, model_path)
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"] = round(cv_scores.std(), 4)
        
        # Average prediction probability (confidence)
        avg_confidence = round(np.mean(np.max(y_pred_proba, axis=1)), 4)
        metrics["avg_confidence"] = avg_confidence
        
        print("\n📈 Model Performance:")
        print(f"   Accuracy: {metrics['accuracy']}")
        print(f"   Precision: {metrics['precision']}")
        print(f"   Recall: {metrics['recall']}")
        print(f"   F1-Score: {metrics['f1_score']}")
        print(f"   CV F1-Score: {metrics['cv_f1_mean']} ± {metrics['cv_f1_std']}")
        print(f"   Average Confidence: {metrics['avg_confidence']}")
        
        # Detailed classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))
        
        # Mapping class distribution
        mapped_distribution = {
            "Malignant": class_distribution.get(0, 0),
            "Benign": class_distribution.get(1, 0)
        }
        
        return {
            "error": False,
            "message": "Model trained successfully",
            "data": {
                "metrics": metrics,
                "class_distribution": mapped_distribution,
                "best_params": grid_search.best_params_,
                "model_info": {
                    "n_layers": len(best_model.hidden_layer_sizes),
                    "total_params": sum(best_model.hidden_layer_sizes),
                    "n_iter": best_model.n_iter_,
                    "loss": round(best_model.loss_, 6) if hasattr(best_model, 'loss_') else None
                }
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

# Alternative: Quick training without grid search for faster iteration
def train_quick_model(model_path="ml/model.pkl", scaler_path="ml/scaler.pkl"):
    """
    Faster training with optimized parameters based on common breast cancer dataset patterns
    """
    try:
        df = load_training_data()
        X = df.drop("diagnosis", axis=1)
        y = LabelEncoder().fit_transform(df["diagnosis"])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dump(scaler, scaler_path)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE if needed
        unique, counts = np.unique(y_train, return_counts=True)
        if min(counts) / max(counts) < 0.7:
            sm = SMOTE(random_state=42, k_neighbors=3)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        
        # Optimized model parameters for breast cancer classification
        model = MLPClassifier(
            hidden_layer_sizes=(150, 100, 50),  # Deeper network
            activation='relu',
            solver='adam',
            alpha=0.001,  # Slightly higher regularization
            learning_rate_init=0.0005,
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            tol=1e-5,  # Lower tolerance for better convergence
            random_state=42,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        dump(model, model_path)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "avg_confidence": round(np.mean(np.max(y_pred_proba, axis=1)), 4)
        }
        
        return {
            "error": False,
            "message": "Quick model trained successfully",
            "data": {
                "metrics": metrics,
                "model_info": {
                    "n_iter": model.n_iter_,
                    "loss": round(model.loss_, 6)
                }
            }
        }
        
    except Exception as e:
        return {
            "error": True,
            "message": "Quick training failed",
            "data": {"error_message": str(e)}
        }