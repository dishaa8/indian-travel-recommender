import streamlit as st
import pandas as pd
import joblib
import numpy as np
from recommender_module import load_data, load_model, load_tag_similarity, recommend_cities_ml

st.set_page_config(page_title="India Travel Recommender", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF6B35, #F7931E);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.sub-header {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.destination-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.destination-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}
.climate-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 0.25rem;
}
.tropical { background-color: #d4edda; color: #155724; }
.moderate { background-color: #cce5ff; color: #004085; }
.cold { background-color: #e2e3e5; color: #383d41; }
.extreme { background-color: #f8d7da; color: #721c24; }
.hot { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

df = load_data()
model = load_model()
co_matrix = load_tag_similarity()

st.markdown('<h1 class="main-header">India Travel Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find your perfect destination using filters or our ML recommender!</p>', unsafe_allow_html=True)

st.sidebar.header("🎯 Find Your Perfect Trip")
budget_type = st.sidebar.selectbox("💰 Budget Category", ["Budget", "Midrange", "Luxury"])
budget_mapping = {
    "Budget": "Budget_Per_Day",
    "Midrange": "Midrange_Per_Day",
    "Luxury": "Luxury_Per_Day"
}
selected_budget_col = budget_mapping[budget_type]
max_budget = st.sidebar.slider("Max Daily Budget (₹)",
    min_value=int(df[selected_budget_col].min()),
    max_value=int(df[selected_budget_col].max()),
    value=int(df[selected_budget_col].quantile(0.7)),
    step=100
)
climate_options = ["All"] + sorted(df['Climate'].dropna().unique().tolist())
selected_climate = st.sidebar.selectbox("🌡️ Preferred Climate", climate_options)
df['Tags_List'] = df['Tags'].apply(lambda x: [tag.strip().lower() for tag in x] if isinstance(x, list) else [])
all_tags = sorted(set(tag for tags in df['Tags_List'] for tag in tags))
selected_tags = st.sidebar.multiselect("🎯 Preferred Tags", all_tags)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
travel_month = st.sidebar.selectbox("🗓️ Travel Month", months)

filtered_df = df[df[selected_budget_col] <= max_budget].copy()
if selected_climate != "All":
    filtered_df = filtered_df[filtered_df['Climate'] == selected_climate]
if selected_tags:
    filtered_df = filtered_df[filtered_df['Tags_List'].apply(lambda tags: any(tag in tags for tag in selected_tags))]
filtered_df = filtered_df.sort_values(selected_budget_col)

tab1, tab2 = st.tabs(["Top Destinations", "ML Recommendations"])

with tab1:
    st.subheader(f"🏆 Top Destinations ({len(filtered_df)})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>{len(filtered_df)}</h3><p>Destinations Found</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>₹{filtered_df[selected_budget_col].mean():,.0f}</h3><p>Average Cost</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>₹{filtered_df[selected_budget_col].min():,.0f}</h3><p>Cheapest</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>{filtered_df['State'].nunique()}</h3><p>States</p></div>", unsafe_allow_html=True)

    for _, row in filtered_df.iterrows():
        climate_class = str(row['Climate']).lower() if pd.notna(row['Climate']) else 'unknown'
        st.markdown(f"""
        <div class="destination-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <h3 style="margin: 0; color: #2c3e50;">🏛️ {row['Destination']} ({row['State']})</h3>
                    <p><span class="climate-badge {climate_class}">{row['Climate']}</span></p>
                    <p><strong>Best Season:</strong> {row['Best_Season']}</p>
                    <p><strong>Tags:</strong> {", ".join(row['Tags'])}</p>
                </div>
                <div style="text-align: right;">
                    <h2 style="color: #27ae60;">₹{row[selected_budget_col]:,}</h2>
                    <p>per day</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("🤖 ML Model: Destination-Based Recommendations")

    if model is None or co_matrix is None:
        st.warning("ML model not available. Please train the model first.")
    else:
        selected_city = st.selectbox("Select a city you like:", df['Destination'].tolist())

        city_row = df[df['Destination'] == selected_city].iloc[0]
        user_climate = city_row['Climate']
        user_tags = city_row['Tags']
        user_budget = city_row[selected_budget_col]
        user_best_season = city_row['Best_Season']
        user_state = city_row['State']

        travel_month = user_best_season.split('-')[0] if pd.notna(user_best_season) else 'Jan'

        if st.button("Generate Similar Cities"):
            recommendations = recommend_cities_ml(
                df, model, co_matrix,
                user_climate, user_tags, user_budget,
                budget_type, travel_month, user_state
            )
            recommendations = recommendations[recommendations['Destination'] != selected_city]
            recommendations = recommendations.sort_values(by='Similarity', ascending=False)

            if recommendations.empty:
                st.info("No similar destinations found.")
            else:
                for _, row in recommendations.iterrows():
                    climate_class = str(row['Climate']).lower() if pd.notna(row['Climate']) else 'unknown'
                    st.markdown(f"""
                    <div class="destination-card">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div style="flex: 1;">
                                <h3 style="margin: 0; color: #2c3e50;">🏛️ {row['Destination']} ({row['State']})</h3>
                                <p><span class="climate-badge {climate_class}">{row['Climate']}</span></p>
                                <p><strong>Best Season:</strong> {row['Best_Season']}</p>
                                <p><strong>Tags:</strong> {", ".join(row['Tags'])}</p>
                                <p><small>🔍 Similarity Breakdown — Tag: {row['tag_match']:.2f}, Budget: {row['budget_match']:.2f}, Season: {row['season_match']:.2f}, State: {row['state_match']:.2f}</small></p>
                            </div>
                            <div style="text-align: right;">
                                <h2 style="color: #27ae60;">₹{row[selected_budget_col]:,}</h2>
                                <p>per day</p>
                                <p><small>🌟 Final Similarity Score: {row['Similarity']:.2f}</small></p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)