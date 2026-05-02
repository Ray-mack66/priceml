import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import joblib

df = pd.read_csv("data/car_prices.csv", header=None)
df.columns = [
    "symboling", "normalized_losses", "make", "fuel_type",
    "aspiration", "num_doors", "body_style", "drive_wheels",
    "engine_location", "wheel_base", "length", "width", "height",
    "curb_weight", "engine_type", "num_cylinders", "engine_size",
    "fuel_system", "bore", "stroke", "compression_ratio",
    "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"
]
df = df[df["price"] != "price"].reset_index(drop=True)
df.replace("?", np.nan, inplace=True)
num_cols = ["horsepower", "peak_rpm", "price", "engine_size",
            "curb_weight", "highway_mpg", "city_mpg",
            "wheel_base", "length", "width"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(subset=["price"], inplace=True)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print("Shape:", df.shape)

features = ["engine_size", "curb_weight", "horsepower",
            "highway_mpg", "city_mpg", "wheel_base", "length", "width"]
X = df[features]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"R² : {r2:.2f}")
print(f"RMSE : {rmse:.0f} $")
joblib.dump(model, "model/model.pkl")
joblib.dump(features, "model/features.pkl")
print("Modèle sauvegardé ✅")
# ── 8. Clustering K-Means ──────────────────────
cluster_features = ["price", "engine_size", "horsepower"]
X_cluster = df[cluster_features].copy()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_cluster)

joblib.dump(kmeans, "model/kmeans.pkl")
joblib.dump(cluster_features, "model/cluster_features.pkl")

print("\nSegments K-Means :")
print(df.groupby("segment")[["price", "horsepower", "engine_size"]].mean().round(0))