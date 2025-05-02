# Pair10

1. Örnek Soru
Sipariş Verme Alışkanlığı Tahmini

Northwind veritabanında müşterilerin toplam harcaması, sipariş sayısı ve ortalama sipariş büyüklüğüne göre bir müşterinin önümüzdeki 6 ay içinde tekrar sipariş verip vermeyeceğini tahmin eden bir derin öğrenme modeli kur.

İpucu:
Veritabanından Orders, Order Details, Customers tablolarını kullan.

"Son sipariş tarihi" bilgisine göre 6 ay sınırı belirle.

Ar-Ge Konuları:
Temporal Features: Mevsimsellik etkisi var mı? (Örn: Yaz aylarında sipariş artıyor mu?)

Data Augmentation: Müşteri datasını arttırarak daha büyük bir veri seti oluşturup modelin başarısını gözlemle.

Class Imbalance: Eğer az kişi sipariş veriyorsa, class_weight veya SMOTE gibi yöntemlerle çözüm üret.

2. Örnek Soru
Ürün İade Risk Skoru

Müşterilerin daha önceki siparişlerindeki indirim oranı, ürün miktarı ve harcama miktarına göre bir siparişin iade edilme riskini tahmin eden bir derin öğrenme modeli oluştur.

İpucu:
Order Details tablosunda Discount bilgisi var.

(Northwind küçük olduğu için) İade olayını yüksek indirim + düşük harcama gibi bir mantıkla sahte etiketleyebilirsin.

Ar-Ge Konuları:
Cost-sensitive Learning: İade edilen ürünlerin firmaya maliyeti daha yüksek. Modeli bu durumu daha ciddiye alacak şekilde ağırlıklandır.

Explainable AI (XAI): SHAP veya LIME gibi yöntemlerle "Model neden bu siparişi riskli buldu?" açıklamasını çıkar.

3. Örnek Soru
Yeni Ürün Satın Alma Potansiyeli

Müşterilerin geçmiş satın alma kategorilerine (örneğin "Beverages", "Confections") bakarak, yeni çıkan bir ürünü satın alma ihtimallerini tahmin eden bir sinir ağı modeli geliştir.

İpucu:
Products, Categories, Order Details ve Orders tablolarını birleştir.

Müşterinin hangi kategorilerde ne kadar harcama yaptığı gibi özellikler üret.

Ar-Ge Konuları:
Recommendation Systems: Deep Learning tabanlı ürün öneri sistemleri araştır (örneğin Neural Collaborative Filtering, AutoEncoders).

Multi-label Prediction: Aynı anda birkaç ürünü birden önerebilecek bir sistem geliştir.

****Tüm örnekleri api haline getiriniz.

***Cumartesi sunulacak
***20 dk ile sınırlı, 20 dk dolduğunda sunum duracak.
