# Prospect Pulse: AI-Driven Lead Prioritization Pipeline

**Prospect Pulse** is an end-to-end Machine Learning solution designed to transform how sales teams engage with potential clients. Instead of navigating a random list, this system uses predictive analytics to rank prospects by their likelihood to convert.

## 🚀 The Business Problem

Sales organizations often waste a significant amount of time on "cold" leads. This project solves that inefficiency by providing a **data-driven priority list**, ensuring that high-value targets are contacted first, significantly increasing the conversion rate per outreach hour.

## 🛠️ System Architecture

The project is built as a modular pipeline to ensure stability and ease of deployment in a Linux environment:

- **Training Engine (`train.py`):** Establishes the model baseline using historical conversion data.
- **Scoring Engine (`scoring_engine.py`):** A batch processor that applies the model to new, unseen prospects to generate probability scores.
- **Interactive Dashboard (`app.py`):** A Streamlit-based interface where sales reps view the ranked list and log outreach outcomes.
- **Feedback Loop (`retrain.py`):** A "learning" script that merges real-world feedback into the training set for continuous model improvement.

## 💻 Tech Stack

- **Language:** Python
- **ML Library:** Scikit-Learn (Logistic Regression)
- **Interface:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Serialization:** Joblib

## 🔄 The Feedback Loop (Human-in-the-Loop)

Unlike static models, Prospect Pulse features a **Continuous Learning** mechanism. When a user logs an outreach result (Interested/Not Interested) via the dashboard, the data is captured in a feedback log. The `retrain.py` script automatically incorporates this new data, allowing the model to adapt to changing market trends without manual code updates.

## 📋 How to Run

1. **Initialize the Model:** `python train.py`
2. **Generate Scored List:** `python scoring_engine.py`
3. **Launch the Dashboard:** `streamlit run app.py`
4. **Update the Model:** `python retrain.py` (Best scheduled as a weekly Cron Job)

