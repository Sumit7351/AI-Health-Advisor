
import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "dataset.csv"
MODEL_PATH = "best_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
EXPLAINER_PATH = "shap_explainer.pkl"
REPORT_PATH = r"C:\Users\Sumit\.gemini\antigravity\brain\a15bdc4b-0e1d-44d6-ba3c-3544de583025\model_performance_report.md"

def train_and_select_model():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    
    # Drop rows with NaNs if any
    df = df.dropna()
    
    X = df.drop("Disease", axis=1)
    y = df["Disease"]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "MLP Classifier": MLPClassifier(max_iter=500, random_state=42)
    }
    
    results = {}
    print(f"\nTraining and evaluating {len(models)} models...")
    
    report_content = "# Model Performance Comparison\n\n| Model | Accuracy | F1-Score |\n|-------|----------|----------|\n"
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {"model": model, "accuracy": acc, "f1": f1}
        print(f"  -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        report_content += f"| {name} | {acc:.4f} | {f1:.4f} |\n"
        
    # Select best model
    # Prioritize Tree-based models for better interpretability (SHAP) IF performance is close
    preferred_models = ["Random Forest", "Gradient Boosting", "Extra Trees"]
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_f1 = results[best_name]["f1"]
    
    # Check if a preferred model is within 1% of the best
    current_best_model = best_name
    for pm in preferred_models:
        if pm in results:
            if results[pm]["f1"] >= (best_f1 - 0.01):
                # Switch to preferred model if it's competitive
                current_best_model = pm
                # If we found a preferred model that matches best performance, break (prioritize order in preferred_models)
                break
    
    best_name = current_best_model # Use the selected (possibly preferred) model
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]
    
    print(f"\n🏆 Best Model Selected: {best_name} with Accuracy: {best_acc:.4f}")
    report_content += f"\n\n**Selected Model:** {best_name} (Optimized for Accuracy & Interpretability)"
    
    # Save report artifact directly
    try:
        with open(REPORT_PATH, "w") as f:
            f.write(report_content)
        print(f"Report saved to {REPORT_PATH}")
    except Exception as e:
        print(f"Could not save report artifact: {e}")
    
    # Save artifacts
    print("Saving model and artifacts...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    # Initialize and save SHAP explainer
    print("Initializing SHAP explainer...")
    explainer = None
    
    # Use TreeExplainer for tree models (FAST and EXACT)
    if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, ExtraTreesClassifier)):
        print("Using TreeExplainer")
        explainer = shap.TreeExplainer(best_model)
    # Use KernelExplainer for others (SLOW, needs background data summary)
    else:
        print("Using KernelExplainer with K-Means summary")
        # Summarize background data to 50 representative samples to speed up inference
        # K-means typically works for dense data; for sparse/categorical 0/1 data, median or random sample might be safer,
        # but kmeans is standard for tabular.
        # Ensure we convert to numpy float for safety
        background_summary = shap.kmeans(X_train.values.astype(float), 50) 
        
        # Kernel explainer needs a predict function that returns probabilities
        explainer = shap.KernelExplainer(best_model.predict_proba, background_summary)
    
    joblib.dump(explainer, EXPLAINER_PATH)
    print("Training complete!")
    
    return best_name, best_acc

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run training first.")
    
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    return model, le, explainer

def predict_disease(symptoms_dict):
    """
    symptoms_dict: dict of {symptom_name: 1/0}
    """
    model, le, explainer = load_artifacts()
    
    df = pd.read_csv(DATA_PATH)
    feature_names = df.drop("Disease", axis=1).columns.tolist()
    
    input_data = pd.DataFrame([symptoms_dict], columns=feature_names).fillna(0)
    
    # Predict
    prediction_idx = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data).max()
    disease = le.inverse_transform([prediction_idx])[0]
    
    # Explain
    shap_values = explainer.shap_values(input_data, check_additivity=False)
    
    # Handle SHAP output format (it varies by model type)
    if isinstance(shap_values, list):
        class_shap_values = shap_values[prediction_idx]
    else:
        # Binary case or TreeExplainer sometimes returns differently
        if len(shap_values.shape) == 3: # (samples, features, classes)
             class_shap_values = shap_values[:, :, prediction_idx]
        else:
             class_shap_values = shap_values
        
    vals = np.array(class_shap_values[0]).flatten()
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    
    # Filter to only show symptoms that the user actually has (value == 1)
    current_symptoms = input_data.iloc[0]
    present_symptoms = current_symptoms[current_symptoms == 1].index.tolist()
    
    # Also show top factors regardless of presence if they are strongly POSITIVE contributions
    # But usually user wants to know which of THEIR symptoms matters.
    feature_importance = feature_importance[feature_importance['col_name'].isin(present_symptoms)]
    
    feature_importance['feature_importance_vals'] = pd.to_numeric(feature_importance['feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_features = feature_importance.head(5).to_dict(orient='records')
    
    return {
        "disease": disease,
        "confidence": float(prediction_proba),
        "explanation": top_features
    }

if __name__ == "__main__":
    train_and_select_model()
