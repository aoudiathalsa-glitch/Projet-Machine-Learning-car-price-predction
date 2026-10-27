import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

# ==============================
# CONFIGURATION PAGE
# ==============================

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ==============================
# STYLE (CORRECTION VISIBILITÉ)
# ==============================

st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: black;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# TITRE
# ==============================

st.title("Used Cars Price Prediction")
st.subheader("Machine Learning Dashboard")

# ==============================
# CHARGEMENT DES DONNÉES
# ==============================

df = pd.read_csv("car data.csv")
df.columns = df.columns.str.lower()

# Feature engineering
df['age'] = 2026 - df['year']

# Suppression colonne inutile
df = df.drop(['car_name'], axis=1)

# Encodage
df = pd.get_dummies(df, drop_first=True)

# Séparation X / y
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# ENTRAÎNEMENT DES MODÈLES
# ==============================

linear = LinearRegression()
linear.fit(X, y)

ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

svr = SVR(kernel='rbf', C=100, gamma=0.01)
svr.fit(X_scaled, y)

# ==============================
# SIDEBAR INPUT
# ==============================

st.sidebar.header("Enter Car Details")

year = st.sidebar.slider("Year", 2000, 2025, 2015)
present_price = st.sidebar.slider("Present Price", 0.0, 50.0, 5.0)
kms_driven = st.sidebar.slider("Kms Driven", 0, 200000, 50000)
owner = st.sidebar.slider("Owner", 0, 3, 0)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# Feature calculée
age = 2026 - year

# ==============================
# CRÉATION INPUT
# ==============================

input_dict = {
    'year': year,
    'present_price': present_price,
    'kms_driven': kms_driven,
    'owner': owner,
    'age': age,
    'fuel_type_diesel': 1 if fuel == "Diesel" else 0,
    'seller_type_individual': 1 if seller == "Individual" else 0,
    'transmission_manual': 1 if transmission == "Manual" else 0
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_df)

# ==============================
# PRÉDICTIONS
# ==============================

pred_linear = linear.predict(input_df)[0]
pred_ridge = ridge.predict(input_scaled)[0]
pred_svr = svr.predict(input_scaled)[0]

# ==============================
# AFFICHAGE RÉSULTATS
# ==============================

st.subheader("Prediction Results")

col1, col2, col3 = st.columns(3)

col1.metric("Linear Regression", f"{pred_linear:.2f}")
col2.metric("Ridge (Best Model)", f"{pred_ridge:.2f}")
col3.metric("SVR", f"{pred_svr:.2f}")

# Titre de la section
st.subheader("Final Predicted Price")

# Affiche le prix prédit par le modèle Ridge (le meilleur modèle)
# round(pred_ridge, 2) = arrondir à 2 chiffres après la virgule
# "lakhs" = unité utilisée dans le dataset
st.write("Estimated price (Ridge model):", round(pred_ridge, 2), "lakhs")

# Conversion du prix en euros
# 1 lakh ≈ 1100 euros (approximation)
price_euro = pred_ridge * 1100

# Affiche le prix converti en euros
st.write("Estimated price in euros:", round(price_euro, 2), "€")

st.success("Best model based on R2: Ridge Regression")

# ==============================
# GRAPHE COMPARAISON
# ==============================

st.subheader("Model Comparison")

models = ["Linear", "Ridge", "SVR"]
values = [pred_linear, pred_ridge, pred_svr]

plt.figure()
plt.bar(models, values)
plt.title("Predicted Price Comparison")
plt.xlabel("Models")
plt.ylabel("Price")

st.pyplot(plt)

# ==============================
# INTERPRÉTATION
# ==============================

avg_price = np.mean(values)

st.subheader("Interpretation")

if avg_price > 10:
    st.write("High price vehicle")
elif avg_price > 5:
    st.write("Medium price vehicle")
else:
    st.write("Low price vehicle")

# ==============================
# AFFICHER DATASET
# ==============================

if st.checkbox("Show dataset"):
    st.write(df.head())
    # ==============================
# CONFIGURATION PAGE
# ==============================

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ==============================
# STYLE (IMPORTANT)
# ==============================

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    color: black !important;
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)