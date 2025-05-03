import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import tensorflow as tf
import uvicorn

# PostgreSQL bağlantısı
engine = create_engine("postgresql+psycopg2://postgres:sifre@localhost:5432/database_name")

app = FastAPI(title="Dairy Products Purchase Prediction API")

class CustomerRequest(BaseModel):
    customer_id: str

print("Model ve ölçekleyici yükleniyor...")

# SQL sorgusu
df = pd.read_sql_query("""
WITH customer_category_stats AS (
  SELECT
    o.customer_id,
    cat.category_name,
    SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_spent,
    COUNT(DISTINCT o.order_id) AS order_count
  FROM orders o
  JOIN order_details od ON o.order_id = od.order_id
  JOIN products p ON od.product_id = p.product_id
  JOIN categories cat ON p.category_id = cat.category_id
  GROUP BY o.customer_id, cat.category_name
)
SELECT
  customer_id,
  category_name,
  total_spent,
  order_count
FROM customer_category_stats
ORDER BY customer_id, category_name;
""", engine)
print("df:")
print(df.head())

spent = df.pivot_table(index='customer_id', columns='category_name', values='total_spent', aggfunc='sum', fill_value=0)
count = df.pivot_table(index='customer_id', columns='category_name', values='order_count', aggfunc='sum', fill_value=0)
print("Spent:")
print(spent.head())
print("Count:")
print(count.head())

spent.columns = [f"spent_{cat.lower()}" for cat in spent.columns]
count.columns = [f"count_{cat.lower()}" for cat in count.columns]
wide_df = pd.concat([spent, count], axis=1).reset_index()

print("New version:")
print(spent.head())
print(count.head())

print("Wide DataFrame:")
print(wide_df.head())
# Etiket oluştur
wide_df['label'] = (wide_df['spent_dairy products'] > 0).astype(int)

# X, y
X = wide_df.drop(['customer_id', 'label', 'spent_dairy products', 'count_dairy products'], axis=1)
y = wide_df['label'].values

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veri dengeleme - SMOTE
print("Orijinal sınıf dağılımı:", Counter(y))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("SMOTE sonrası sınıf dağılımı:", Counter(y_resampled))

# Model oluştur
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_resampled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(X_resampled, y_resampled, epochs=30, batch_size=32, verbose=1)

# Performans
preds = (model.predict(X_scaled) > 0.5).astype(int).flatten()
cm = confusion_matrix(y, preds)
print("Confusion Matrix:\n", cm)

print("Model ve scaler hazır!")

@app.post("/predict")
async def predict_dairy_purchase(req: CustomerRequest):
    cust_id = req.customer_id
    if cust_id not in wide_df['customer_id'].values:
        raise HTTPException(status_code=404, detail="Customer ID not found")

    row = wide_df[wide_df['customer_id'] == cust_id].drop(['customer_id', 'label', 'spent_dairy products', 'count_dairy products'], axis=1)
    row_scaled = scaler.transform(row)
    prob = model.predict(row_scaled).flatten()[0]

    return {
        "customer_id": cust_id,
        "purchase_likelihood_dairy_products": float(prob),
        "prediction": int(prob > 0.5)
    }

if __name__ == "__main__":
    uvicorn.run("newproduct:app", host="127.0.0.1", port=8000, reload=False)
