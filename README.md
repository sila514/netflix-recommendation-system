# 🎬 Netflix Benzeri İçerik Tabanlı Öneri Sistemi

İçerik tabanlı (content-based) filtreleme kullanarak film ve dizi önerileri sunan bir Python projesi.

## 🎯 Projenin Amacı

Verilen bir film veya diziye benzer içerikleri, metin analizi teknikleri kullanarak önermek. Makine öğrenmesi temellerini ve öneri sistemlerinin çalışma prensiplerini göstermektedir.

## 🛠 Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|-----------|---------------|
| **Python 3** | Ana programlama dili |
| **Pandas** | Veri yükleme ve ön işleme |
| **Scikit-learn** | TF-IDF vektörizasyonu ve Kosinüs Benzerliği |

## 📖 Nasıl Çalışır?

1. **TF-IDF Vectorization**: Film açıklamaları ve türleri sayısal vektörlere dönüştürülür
2. **Cosine Similarity**: Tüm içerik çiftleri arasındaki benzerlik hesaplanır
3. **Öneri**: Girilen başlığa en yakın 5 içerik sıralanarak sunulur

## 🚀 Çalıştırma

```bash
pip install pandas scikit-learn
python recommender.py
```

Program başlatıldığında mevcut içerikleri listeler ve terminal üzerinden film/dizi adı girmenizi bekler.

## 📂 Dosya Yapısı

```
├── recommender.py   # Ana öneri sistemi kodu
├── README.md        # Proje dokümantasyonu
```

## 📝 Örnek Çıktı

```
Film/dizi adı girin: Breaking Bad

🎯 'Breaking Bad' için öneriler:
  1. Ozark           (benzerlik: 0.25)
  2. Narcos          (benzerlik: 0.22)
  3. Mindhunter      (benzerlik: 0.18)
  ...
```
