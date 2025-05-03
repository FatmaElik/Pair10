# Yeni Ürün Satın Alma Potansiyeli

Bu proje, müşterilerin geçmiş satın alma verilerine dayanarak süt ürünleri satın alma olasılığını tahmin eden bir makine öğrenmesi modeli uygular. Model FastAPI endpoint'i üzerinden hizmet verir.

## Özellikler

- TensorFlow kullanılarak geliştirilmiş derin öğrenme modeli
- Dengesiz verilerle başa çıkmak için SMOTE kullanımı
- FastAPI endpoint ile tahmin servisi
- PostgreSQL veritabanı entegrasyonu
- StandardScaler ile özellik normalizasyonu

## Proje Yapısı

- `newproduct.py`: Veri ön işleme, model eğitimi ve API servisi için ana betik
- `requirements.txt`: Proje bağımlılıklarını içeren dosya

## Kullanılan Özellikler

### Müşteri Kategorisi Düzeyinde Özellikler:
- Kategori bazında toplam harcama
- Kategori bazında sipariş sayısı
- Kategori bazında ortalama harcama

## Hedef Değişken

Hedef değişken (label) aşağıdaki koşula göre oluşturulmuştur:
- Müşterinin süt ürünleri kategorisinde harcama yapıp yapmadığı (binary)

## Kurulum

1. Sanal ortam oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. PostgreSQL veritabanını ayarlayın:
- Gerekirse `newproduct.py` dosyasındaki bağlantı dizesini güncelleyin:
  ```python
  engine = create_engine("postgresql+psycopg2://postgres:sifre@localhost:5432/database_name")
  ```

## Kullanım

1. FastAPI sunucusunu başlatın:
```bash
python newproduct.py
```

2. API şu adreste kullanılabilir olacaktır: `http://127.0.0.1:8000`

3. `/predict` endpoint'i ile tahmin yapın:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"customer_id": "müşteri_id"}'
```

## API Endpoint'leri

### POST /predict
Müşterinin süt ürünleri satın alma olasılığını tahmin eder.

İstek gövdesi:
```json
{
    "customer_id": "string"
}
```

Yanıt:
```json
{
    "customer_id": "string",
    "purchase_likelihood_dairy_products": float,
    "prediction": int
}
```

## Model Detayları

Model şunları kullanır:
- Çoklu yoğun katmanlara sahip sinir ağı mimarisi
- Sınıf dengesizliğini gidermek için SMOTE
- Özellik normalizasyonu için StandardScaler
- Sigmoid aktivasyonlu ikili sınıflandırma

## Lisans

[Lisans bilgilerinizi buraya ekleyin] 