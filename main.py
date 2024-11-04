import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def load_data():
    # Tibbiy ma'lumotlar
    data = pd.DataFrame({
        'Yosh': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'Qon bosimi': [120, 130, 115, 140, 135, 125, 145, 150, 155, 160],
        'Qand miqdori': [5.1, 5.5, 5.0, 6.1, 5.9, 5.8, 6.5, 7.0, 7.5, 7.2],
        'Jins': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],  # 1 - erkak, 0 - ayol
        'Kasallik': [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    })
    return data

def select_features(data):
    X = data[['Yosh', 'Qon bosimi', 'Qand miqdori', 'Jins']]
    y = data['Kasallik']
    mi = mutual_info_classif(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi_series

def get_best_features(data):
    X = data[['Yosh', 'Qon bosimi', 'Qand miqdori', 'Jins']]
    y = data['Kasallik']
    mi_series = select_features(data)
    best_score = 0
    best_features = None
    for k in range(1, len(X.columns) + 1):
        selected_features = mi_series.nlargest(k).index
        X_selected = X[selected_features]
        scores = cross_val_score(RandomForestClassifier(), X_selected, y, cv=5)
        avg_score = scores.mean()
        if avg_score > best_score:
            best_score = avg_score
            best_features = selected_features
    return best_features, best_score

if name == "main":
    data = load_data()
    best_features, best_score = get_best_features(data)
    print("Tanlangan eng yaxshi belgilar:", best_features.tolist())
    print("Eng yaxshi aniqlik:", best_score)
