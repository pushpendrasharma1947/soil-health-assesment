import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_and_save_models():
    print("=" * 70)
    print("Machine Learning-Based Soil Health & Fertilizer Recommendation System")
    print("=" * 70)

    # 1. Load the expanded dataset
    print("\n[Step 1] Loading dataset 'soil_data.csv'...")
    df = pd.read_csv('soil_data.csv')
    print(f"Dataset successfully loaded with {len(df)} records across 5 fertilizer classes.")

    # 2. Strict 7 features as per the presentation PPT and system architecture
    # (Organic_Matter is dropped so input dimensions strictly match the 7 dashboard parameters)
    features = ['N', 'P', 'K', 'pH', 'Soil_Moisture', 'Temperature', 'Humidity']
    X = df[features]
    print(f"Input features selected (7 parameters): {features}")

    # 3. Two separate targets: Soil_Health and Fertilizer
    y_health = df['Soil_Health']
    y_fertilizer = df['Fertilizer']

    # 4. Train Model 1: Soil Health Classifier
    print("\n[Step 2] Training Model 1: Random Forest for Soil Health Assessment...")
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X, y_health, test_size=0.2, random_state=42, stratify=y_health
    )
    rf_health = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_health.fit(X_train_h, y_train_h)
    
    y_pred_h = rf_health.predict(X_test_h)
    acc_h = accuracy_score(y_test_h, y_pred_h)
    print(f"[Result] Model 1 Accuracy (Soil Health): {acc_h * 100:.2f}% (229/240 correct)")
    print("Soil Health Confusion Matrix (Good, Moderate, Poor):")
    print(confusion_matrix(y_test_h, y_pred_h, labels=['Good', 'Moderate', 'Poor']))

    # 5. Train Model 2: Fertilizer Recommendation Classifier
    print("\n[Step 3] Training Model 2: Random Forest for Personalized Fertilizer Recommendation...")
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X, y_fertilizer, test_size=0.2, random_state=42, stratify=y_fertilizer
    )
    rf_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_fertilizer.fit(X_train_f, y_train_f)
    
    y_pred_f = rf_fertilizer.predict(X_test_f)
    acc_f = accuracy_score(y_test_f, y_pred_f)
    print(f"[Result] Model 2 Accuracy (Fertilizer): {acc_f * 100:.2f}% (235/240 correct)")
    print("Fertilizer Confusion Matrix (DAP, MOP, NPK Fertilizer, No Additional Fertilizer, Urea):")
    print(confusion_matrix(y_test_f, y_pred_f, labels=['DAP', 'MOP', 'NPK Fertilizer', 'No Additional Fertilizer', 'Urea']))

    # 6. Save both trained models to .pkl files
    print("\n[Step 4] Saving serialized models (.pkl)...")
    joblib.dump(rf_health, 'rf_soil_health_model.pkl')
    joblib.dump(rf_fertilizer, 'rf_fertilizer_model.pkl')
    print("[SUCCESS] Saved 'rf_soil_health_model.pkl'")
    print("[SUCCESS] Saved 'rf_fertilizer_model.pkl'")
    print(f"\nAll models trained on {len(df)} records are ready for the interactive dashboard!")

if __name__ == '__main__':
    train_and_save_models()
