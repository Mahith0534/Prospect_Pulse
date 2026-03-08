import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Sales Portal", layout="wide")
st.title("🚀 Today's Prioritized Prospects")

# 1. Load the Ranked List
@st.cache_data
def load_data():
    return pd.read_csv('data/daily_priority_list.csv')

df = load_data()

# 2. Display the sorted table
st.subheader("High-Intent Leads")
st.dataframe(df[['Company_Name', 'Industry', 'Conversion_Probability']].head(20))

# 3. THE FEEDBACK LOOP
st.write("---")
st.subheader("🎯 Log Outreach Result")

with st.form("outreach_feedback"):
    # 1. Select the company from the current prioritized list
    target_company = st.selectbox("Which company did you call?", df['Company_Name'].tolist())
    
    # 2. Capture the actual outcome
    result = st.radio("Outcome", ["Interested (Lead Converted)", "Not Interested"])
    submit = st.form_submit_button("Log Feedback & Update Database")
    
    if submit:
        # 3. Find the data for this specific company from our current list
        company_data = df[df['Company_Name'] == target_company].iloc[0].to_dict()
        
        # 4. Map the human feedback to a 1 or 0 for the model
        company_data['Converted'] = 1 if result == "Interested (Lead Converted)" else 0
        
        # 5. Save this to a feedback file (Append mode)
        feedback_file = 'data/prospect_feedback_log.csv'
        feedback_df = pd.DataFrame([company_data])
        
        # If file doesn't exist, write with header; if it does, append without header
        file_exists = os.path.isfile(feedback_file)
        feedback_df.to_csv(feedback_file, mode='a', index=False, header=not file_exists)
        
        st.success(f"✅ Data for {target_company} has been saved to the Feedback Log for retraining!")