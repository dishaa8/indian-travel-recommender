import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

month_to_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def get_month_similarity(user_month, city_best_season):
    if not city_best_season or pd.isna(city_best_season):
        return 0.0
    try:
        start, end = [month_to_num[m] for m in city_best_season.split('-')]
        user = month_to_num[user_month]
        if start <= end and start <= user <= end:
            return 1.0
        if start > end and (user >= start or user <= end):
            return 1.0
        return 0.5  
    except:
        return 0.0

def load_data():
    df = pd.read_csv("finaldestinations.csv")
    df.columns = df.columns.str.strip()
    df['Tags'] = df['Tags'].apply(lambda x: [tag.strip().lower() for tag in x.split(',')])
    df['Tags_List'] = df['Tags']
    df['Climate'] = df['Climate'].str.lower()
    return df

def load_model():
    return joblib.load("travel_model.pkl") if os.path.exists("travel_model.pkl") else None

def load_tag_similarity():
    return joblib.load("tag_similarity.pkl") if os.path.exists("tag_similarity.pkl") else None

def build_tag_cooccurrence_matrix(df):
    all_tags = sorted(set(tag for tags in df['Tags_List'] for tag in tags))
    tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}
    co_matrix = np.zeros((len(all_tags), len(all_tags)))
    for tags in df['Tags_List']:
        for i in range(len(tags)):
            for j in range(len(tags)):
                if i != j:
                    co_matrix[tag_to_idx[tags[i]], tag_to_idx[tags[j]]] += 1
    row_sums = co_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(co_matrix, row_sums, out=np.zeros_like(co_matrix), where=row_sums != 0)
    return pd.DataFrame(norm_matrix, index=all_tags, columns=all_tags)

def calculate_tag_similarity(user_tags, destination_tags, co_matrix):
    score = 0
    total = 0
    for u in user_tags:
        if u in co_matrix.columns:
            for d in destination_tags:
                if d in co_matrix.columns:
                    score += co_matrix.loc[u, d]
                    total += 1
    return score / total if total > 0 else 0

def cluster_cities(df, n_clusters=5):
    df = df.copy()
    df['Avg_Budget'] = df[['Budget_Per_Day', 'Midrange_Per_Day', 'Luxury_Per_Day']].mean(axis=1)
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(df['Tags_List'])
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    climate_encoded = ohe.fit_transform(df[['Climate']])
    X = np.hstack([tag_matrix, climate_encoded, df[['Avg_Budget']]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Persona'] = kmeans.fit_predict(X)
    return df, kmeans, mlb, ohe

def generate_clustered_user_profiles(df, mlb, ohe, n_users=100):
    users = []
    grouped = df.groupby('Persona')
    for _ in range(n_users):
        persona = np.random.choice(df['Persona'].unique())
        city = grouped.get_group(persona).sample(1).iloc[0]
        tags = np.random.choice(city['Tags_List'], size=np.random.randint(1, 4), replace=False).tolist()
        users.append({
            'climate': city['Climate'],
            'tags': tags,
            'budget_type': np.random.choice(['Budget', 'Midrange', 'Luxury']),
            'budget': int(city[['Budget_Per_Day', 'Midrange_Per_Day', 'Luxury_Per_Day']].mean()),
            'travel_month': np.random.choice(list(month_to_num.keys())),
            'state': city['State']
        })
    return users

def generate_training_data(df, users, co_matrix):
    rows, labels = [], []
    tolerance = {'Budget': 500, 'Midrange': 1000, 'Luxury': 2000}
    for user in users:
        for _, city in df.iterrows():
            row = {
                'user_climate': user['climate'],
                'city_climate': city['Climate'],
                'tag_match': calculate_tag_similarity(user['tags'], city['Tags'], co_matrix),
                'budget_match': 1 if abs(user['budget'] - city[f"{user['budget_type']}_Per_Day"]) <= tolerance[user['budget_type']] else 0,
                'season_match': get_month_similarity(user['travel_month'], city['Best_Season']),
                'state_match': 1 if user['state'] == city['State'] else 0
            }
            sim = (
                0.25 * row['tag_match'] +
                0.25 * row['budget_match'] +
                0.2 * row['season_match'] +
                0.2 * row['state_match']
            )
            noise = np.clip(sim + np.random.normal(0, 0.05), 0, 1)
            rows.append(row)
            labels.append(noise)
    return pd.DataFrame(rows), labels

def recommend_cities_ml(df, model, co_matrix, user_climate, user_tags, user_budget, budget_type, travel_month, user_state):
    import joblib

    climate_ohe = joblib.load("climate_encoder.pkl")
    
    tolerance = {'Budget': 500, 'Midrange': 1000, 'Luxury': 2000}
    data = []

    for _, city in df.iterrows():
        row = {
            'user_climate': user_climate,
            'city_climate': city['Climate'],
            'tag_match': calculate_tag_similarity(user_tags, city['Tags'], co_matrix),
            'budget_match': 1 if abs(user_budget - city[f"{budget_type}_Per_Day"]) <= tolerance[budget_type] else 0,
            'season_match': get_month_similarity(travel_month, city['Best_Season']),
            'state_match': 1 if user_state == city['State'] else 0
        }
        data.append(row)

    X = pd.DataFrame(data)

    climate_encoded = climate_ohe.transform(X[['user_climate', 'city_climate']])
    X_rest = X.drop(columns=['user_climate', 'city_climate']).reset_index(drop=True)
    X_final = pd.concat([pd.DataFrame(climate_encoded), X_rest], axis=1)
    probs = model.predict(X_final)
    df_copy = df.copy()
    df_copy[['tag_match', 'budget_match', 'season_match', 'state_match']] = X[['tag_match', 'budget_match', 'season_match', 'state_match']]
    df_copy['Similarity'] = probs

    return df_copy.sort_values(by='Similarity', ascending=False).head(10)