import pandas as pd
import joblib

def process_new_leads():
    # 1. Load the "Fresh" Leads (Amex hasn't seen these yet)
    # This CSV should have: Company_Name, Industry, Revenue, Web_Visits, etc.
    try:
        new_leads = pd.read_csv('data/new_prospects.csv')
    except FileNotFoundError:
        print("Error: data/new_prospects.csv not found.")
        return

    # 2. Load the Model & Scaler (The 'Brain' from your training)
    model = joblib.load('artifact/model.joblib')
    scaler = joblib.load('artifact/scaler.joblib')
    model_features = joblib.load('artifact/features.joblib')

    # 3. Preprocess the New Leads
    # We must match the 'Industry' encoding used in the training data
    df_encoded = pd.get_dummies(new_leads, columns=['Industry'], drop_first=True)
    
    # Ensure the new list has all the columns the model expects (even if 0)
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    X_new = df_encoded[model_features]
    
    # 4. Generate the Propensity Scores
    X_scaled = scaler.transform(X_new)
    new_leads['Conversion_Probability'] = model.predict_proba(X_scaled)[:, 1]

    # 5. SORT: Highest probability first
    prioritized_list = new_leads.sort_values(by='Conversion_Probability', ascending=False)

    # 6. Save the Final Ranked List for the Sales Team
    prioritized_list.to_csv('data/daily_priority_list.csv', index=False)
    print(f"Successfully prioritized {len(new_leads)} new prospects.")

if __name__ == "__main__":
    process_new_leads()