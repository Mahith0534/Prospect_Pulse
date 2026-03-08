import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model():
    # 1. Load Data
    df = pd.read_csv('data/prospect_pulse_data.csv')

    # 2. Preprocess
    df_encoded = pd.get_dummies(df, columns=['Industry'], drop_first=True)
    X = df_encoded.drop(['Company_ID', 'Converted'], axis=1)
    y = df_encoded['Converted']

    # 3. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 4. Train
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # 5. EXPORT (The Corporate Way)
    # We save these as "artifacts" so the web server can load them instantly
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(list(X.columns), 'features.joblib')
    
    print("Model Artifacts Saved Successfully.")

if __name__ == "__main__":
    train_model()