# train_customer_reorder.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from load_features import load_customer_snapshots


def main():
    # 1) Veri yükle
    df = load_customer_snapshots()

    # 2) Özellikler ve etiket
    X = df[[
        "total_spent",
        "total_orders",
        "avg_order_value",
        "days_since_prev_order",
        "order_month"
    ]].fillna(0).values
    y = df["reorder_6m"].values

    # 3) Eğitim/test bölünümü
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4) Ölçekleme
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 5) Geliştirilmiş MLPClassifier ayarları
    model = MLPClassifier(
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
    model.fit(X_train, y_train)

    # 6) Değerlendirme
    test_acc = model.score(X_test, y_test)
    print(f"Test Doğruluğu: {test_acc:.2f}")

    # 7) Eğitim ve Doğrulama Eğrilerini Görselleştir
    plt.figure()
    plt.plot(model.loss_curve_, label='Eğitim Kayıp')
    if hasattr(model, 'validation_scores_'):
        plt.plot(model.validation_scores_, label='Doğrulama Skoru')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp / Skor')
    plt.title('MLP Eğitimi')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
