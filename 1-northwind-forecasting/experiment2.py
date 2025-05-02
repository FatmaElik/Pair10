# research_experiments.py
"""
Bu script, Ar-Ge konularını denemek için iskelet fonksiyonlar içerir:
- Temporal Özellikler
- Data Augmentation
- Class Imbalance Handling
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from load_features import load_customer_snapshots


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mevsimsellik etkisini yakalamak için 'order_month' dışında ek özellikler ekleyin.
    Örneğin:
      - seasonal_sin, seasonal_cos
      - quarter
    """
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['order_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['order_month'] / 12)
    df['quarter'] = ((df['order_month'] - 1) // 3 + 1)
    return df


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mevcut snapshot'lara küçük gürültü ekleyerek veri setini genişletin.
    Alternatif olarak SMOTE de kullanılabilir.
    """
    df_aug = df.copy()
    noise = 0.05
    for col in ['total_spent', 'avg_order_value']:
        df_aug[col] = df_aug[col] * (1 + np.random.uniform(-noise, noise, size=len(df_aug)))
    return pd.concat([df, df_aug], ignore_index=True)


def handle_class_imbalance(X: pd.DataFrame, y: pd.Series):
    """
    Sınıf dengesizliğini SMOTE ile çözme.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


if __name__ == '__main__':
    df = load_customer_snapshots()
    df_temp = add_temporal_features(df)
    df_aug = augment_data(df_temp)

    # Sadece sayısal özellik sütunlarını seç
    feature_cols = [
        'total_spent', 'total_orders', 'avg_order_value',
        'days_since_prev_order', 'order_month',
        'month_sin', 'month_cos', 'quarter'
    ]
    X = df_aug[feature_cols].fillna(0)
    y = df_aug['reorder_6m']

    # Orijinal sınıf dağılımı
    print("Orijinal sınıf dağılımı:")
    print(y.value_counts())

    # SMOTE ile class imbalance çözümü
    X_res, y_res = handle_class_imbalance(X, y)

    # Örnek büyüklükleri
    print(f"Örnek büyüklük (orijinal):    {len(df_temp)}")
    print(f"Örnek büyüklük (augment edilmiş): {len(df_aug)}")
    print(f"Örnek büyüklük (SMOTE sonrası):  {len(X_res)}")

    # SMOTE sonrası sınıf dağılımı
    print("SMOTE sonrası sınıf dağılımı:")
    print(pd.Series(y_res).value_counts())
