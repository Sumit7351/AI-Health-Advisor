import pandas as pd
import numpy as np
import joblib
import shap
import os
from sklearn.model_selection import train_test_split, cross_val_score
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

def train_and_select_model():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Dataset Info:")
    print(df.info())
    print("Null values:", df.isnull().sum().sum())
    
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
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {"model": model, "accuracy": acc, "f1": f1}
        print(f"  -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
    # Select best model
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]
    
    print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_acc:.4f}")
    
    # Save artifacts
    print("Saving model and artifacts...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    # Initialize and save SHAP explainer
    # Note: SHAP explainers can be large, we'll try to save a lightweight version or just the background data if needed.
    # For Tree models, TreeExplainer is efficient. For others, KernelExplainer is slow.
    # We will prioritize TreeExplainer if the best model is tree-based.
    
    print("Initializing SHAP explainer...")
    explainer = None
    if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, ExtraTreesClassifier)):
        explainer = shap.TreeExplainer(best_model)
    else:
        # Fallback for non-tree models (using a small background sample for speed)
        background = shap.kmeans(X_train, 10)
        explainer = shap.KernelExplainer(best_model.predict_proba, background)
    
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
    
    # Prepare input vector (ensure all columns from training are present)
    # We need to read the columns from the dataset or save them. 
    # For simplicity, we'll read dataset columns again or save feature names.
    # Let's read dataset columns for now as it's available.
    df = pd.read_csv(DATA_PATH)
    feature_names = df.drop("Disease", axis=1).columns.tolist()
    
    input_data = pd.DataFrame([symptoms_dict], columns=feature_names).fillna(0)
    
    # Predict
    prediction_idx = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data).max()
    disease = le.inverse_transform([prediction_idx])[0]
    
    # Explain
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_data)
    
    # Handle SHAP output format (it varies by model type)
    # For multi-class, shap_values is a list of arrays (one for each class)
    if isinstance(shap_values, list):
        # Get shap values for the predicted class
        class_shap_values = shap_values[prediction_idx]
    else:
        # Binary case or different format
        class_shap_values = shap_values
        
    # Get top contributing features
    # class_shap_values is (1, n_features)
    vals = np.array(class_shap_values[0]).flatten()
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    
    # Filter to only show symptoms that the user actually has (value == 1)
    # This aligns with user expectation to see which of *their* symptoms caused the prediction
    current_symptoms = input_data.iloc[0]
    present_symptoms = current_symptoms[current_symptoms == 1].index.tolist()
    feature_importance = feature_importance[feature_importance['col_name'].isin(present_symptoms)]
    
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_features = feature_importance.head(5).to_dict(orient='records')
    
    return {
        "disease": disease,
        "confidence": float(prediction_proba),
        "explanation": top_features
    }

if __name__ == "__main__":
    train_and_select_model()
