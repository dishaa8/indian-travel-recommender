# ML-Based Travel Destination Recommender System

This project is a machine learning-powered travel recommender system built using Python and Streamlit. It suggests Indian travel destinations based on user-selected preferences like budget, season, climate, and tags.

## 🔧 Features

- ML model trained using synthetic user profiles
- Tag similarity via co-occurrence matrix
- Budget, season, and state match scoring
- Explainable recommendations with feature breakdown
- Interactive Streamlit web interface

## 📁 Files Included

- `app.py` – Main Streamlit UI
- `recommender_module.py` – Core ML functions and logic
- `finaldestinations.csv` – Cleaned dataset of 100 destinations
- `travel_model.pkl` – Trained XGBoost model
- `tag_similarity.pkl` – Tag similarity matrix
- `climate_encoder.pkl` – One-hot encoder for climate
- `requirements.txt` – Required Python packages

## 🚀 How to Run

1. Clone this repository  
2. Install dependencies using `pip install -r requirements.txt`  
3. Run the app using:  
   ```
   streamlit run app.py
   ```

## 🌐 Deploy on Streamlit Cloud

- Push all files to a **public GitHub repo**
- Go to https://streamlit.io/cloud and connect your GitHub
- Click "New App", select `app.py` as entrypoint, and deploy

Enjoy your ML-powered travel recommender! ✈️