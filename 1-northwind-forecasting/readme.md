<!-- README.md -->

# Northwind Customer Reorder Prediction

Bu proje, Northwind veritabanındaki müşterilerin “önümüzdeki 6 ay içinde yeniden sipariş verip vermeyeceğini” tahmin eden bir makine öğrenmesi pipeline’ını içerir.

---

## Dosyalar

- **`create_northwind_pg.py`**  
  PostgreSQL üzerinde Northwind şemasını ve tablolarını oluşturur.

- **`load_features.py`**  
  CTE ve pencere fonksiyonları kullanarak müşteri snapshot verilerini (total_spent, total_orders, avg_order_value, days_since_prev_order, order_month) ve 6 aylık yeniden sipariş (`reorder_6m`) etiketini pandas DataFrame olarak döndürür.

- **`train_customer_reorder.py`**  
  `load_features.py`’den aldığı veriyi alır, Scikit-learn’ün `MLPClassifier` modeli ile eğitir, test doğruluğunu raporlar ve eğitim/doğrulama eğrilerini görselleştirir.

- **`research_experiments.py`**  
  Ar-Ge denemeleri için:
  - **Temporal Features:** Mevsimsellik için `month_sin`, `month_cos`, `quarter`  
  - **Data Augmentation:** Küçük gürültü ekleyerek veri setini genişletme  
  - **Class Imbalance:** SMOTE ile dengeli sınıf dağılımı

- **`requirements.txt`**  
  Projede kullanılan Python paketlerini listeler.

---

## Kurulum

1. **Python sanal ortamı oluşturun**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
