# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from load_features import load_customer_snapshots
from imblearn.over_sampling import SMOTE

# AR-GE fonksiyonları
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['order_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['order_month'] / 12)
    df['quarter']   = ((df['order_month'] - 1) // 3 + 1)
    return df

def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    df_aug = df.copy()
    noise = 0.05
    for col in ['total_spent', 'avg_order_value']:
        df_aug[col] = df_aug[col] * (1 + np.random.uniform(-noise, noise, size=len(df_aug)))
    return pd.concat([df, df_aug], ignore_index=True)

def handle_class_imbalance(X: np.ndarray, y: np.ndarray):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

# Özellik listesi
feature_cols = [
    'total_spent','total_orders','avg_order_value',
    'days_since_prev_order','order_month',
    'month_sin','month_cos','quarter'
]

# Input modeli
class FeaturesIn(BaseModel):
    total_spent: float
    total_orders: int
    avg_order_value: float
    days_since_prev_order: int
    order_month: int

# FastAPI uygulaması
app = FastAPI(
    title="Northwind Müşteri Sipariş Tahmin API",
    description="Müşterilerin 6 ay içinde yeniden sipariş verip vermeyeceğini tahmin eden API",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    # 1) Veri yükle ve özellik mühendisliği
    df      = load_customer_snapshots()
    df      = add_temporal_features(df)
    df_aug  = augment_data(df)

    # 2) Özellik ve etiket matrisleri
    X = df_aug[feature_cols].fillna(0).values
    y = df_aug['reorder_6m'].values

    # 3) Sınıf dengesizliğini gider
    X_res, y_res = handle_class_imbalance(X, y)

    # 4) Train/Test split
    X_train, _, y_train, _ = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    # 5) Ölçekleme ve model eğitimi
    app.state.scaler = StandardScaler().fit(X_train)
    X_train_scaled   = app.state.scaler.transform(X_train)
    app.state.model  = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        early_stopping=True,
        validation_fraction=0.1,
        tol=1e-4,
        learning_rate_init=1e-4,
        random_state=42,
        max_iter=300
    )
    app.state.model.fit(X_train_scaled, y_train)

@app.post("/predict")
def predict(features: FeaturesIn):
    # Gelen veriden temporal öznitelikler
    month        = features.order_month
    seasonal_sin = np.sin(2 * np.pi * month / 12)
    seasonal_cos = np.cos(2 * np.pi * month / 12)
    quarter      = (month - 1) // 3 + 1

    x = np.array([[
        features.total_spent,
        features.total_orders,
        features.avg_order_value,
        features.days_since_prev_order,
        features.order_month,
        seasonal_sin,
        seasonal_cos,
        quarter
    ]])
    x_scaled = app.state.scaler.transform(x)
    prob     = app.state.model.predict_proba(x_scaled)[0, 1]
    return {"reorder_probability_6m": float(prob)}

@app.get("/features")
def get_features():
    # Tüm müşteri snapshot verilerini JSON olarak döndür
    df = load_customer_snapshots()
    return df.to_dict(orient="records")
