import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

def update_model():
    # 1. Load the Original Training Data
    base_data = pd.read_csv('data/prospect_pulse_data.csv')
    
    # 2. Check for New Feedback from the Streamlit App
    feedback_file = 'data/prospect_feedback_log.csv'
    
    if os.path.exists(feedback_file):
        feedback_data = pd.read_csv(feedback_file)
        # Combine old data with new human-verified data
        combined_df = pd.concat([base_data, feedback_data], ignore_index=True)
        print(f"Incorporating {len(feedback_data)} new records into the model...")
    else:
        combined_df = base_data
        print("No new feedback found. Re-training on base data.")

    # 3. Preprocess (Same logic as our original engine)
    df_encoded = pd.get_dummies(combined_df, columns=['Industry'], drop_first=True)
    
    # Define features and target
    # Ensure we drop columns like 'Company_Name' or 'Company_ID' if they exist
    X = df_encoded.drop(['Company_ID', 'Converted'], axis=1, errors='ignore')
    if 'Company_Name' in X.columns:
        X = X.drop('Company_Name', axis=1)
    
    y = df_encoded['Converted']

    # 4. Scale and Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)

    # 5. Overwrite the Model Artifacts (The "Update")
    joblib.dump(model, 'artifact/model.joblib')
    joblib.dump(scaler, 'artifact/scaler.joblib')
    joblib.dump(list(X.columns), 'artifact/features.joblib')
    
    print("✅ Model updated successfully with the latest feedback.")

if __name__ == "__main__":
    update_model()