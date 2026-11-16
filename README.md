# Trafik İşareti Görüntüleri ve Sürücü Uyarı Metinlerinin Multimodal Derin Öğrenme ile Füzyonu

**Ders:** Derin Öğrenme (Lisans)  
**Proje Türü:** Multimodal Derin Öğrenme (Görsel + Metin Modaliteleri)  
**Odak Aşaması:** Vize Teslimi — Veri Toplama, Modalite Eşleştirme, Ön İşleme ve Keşifsel Veri Analizi (EDA)

---

## 1. Problem Tanımı ve Multimodal Yapı

### 1.1 Sınıflandırma Görevinin Tanımı
Otonom sürüş ve Gelişmiş Sürücü Destek Sistemleri (ADAS), yol kenarındaki trafik işaretlerini yalnızca "hangi tabela olduğu" düzeyinde tanımakla kalmayıp bu işaretin **sürüş güvenliği açısından aciliyetini ve olası kural ihlali riskini** de doğru sınıflandırmalıdır.

Bu projede, **GTSRB (German Traffic Sign Recognition Benchmark)** veri setinden seçilen **16 temsili trafik işareti alt sınıfı**, sürüş güvenliği ve kural ihlali sonuçlarına göre **4 ana hedef sınıfta** modellenmektedir:

| Sınıf ID | Hedef Sınıf Adı | Risk Seviyesi | Kapsadığı Örnek GTSRB İşaretleri |
| :---: | :--- | :---: | :--- |
| `0` | **Kritik_Yasak_ve_Hiz** | Çok Yüksek | Hız Sınırı 30 (`1`), 50 (`2`), 70 (`4`), Geçme Yasağı (`9`), Girişi Olmayan Yol (`17`) |
| `1` | **Oncelik_ve_Durdurma** | Çok Yüksek | Öncelikli Yol (`12`), Yol Ver (`13`), DUR (`14`) |
| `2` | **Tehlike_ve_Uyari** | Orta-Yüksek | Dikkat/Tehlike (`18`), Yol Çalışması (`25`), Yaya Geçidi (`27`), Buzlanma (`30`) |
| `3` | **Mecburi_Yon_ve_Bilgi** | Düşük-Orta | Sağa Mecburi (`33`), Sola Mecburi (`34`), İleri Mecburi (`35`), Sağdan Gidiniz (`38`) |

> [!NOTE]
> Seçilen 4 ana sınıf ve 16 alt sınıf yapısı; ders kapsamında belirlenen **en az 3 sınıf**, **sınıf başına en az 150 örnek** ve **toplamda en az 500 örnek** veri yeterlilik kurallarını fazlasıyla karşılayacak şekilde tasarlanmıştır.

---

### 1.2 Modalitelerin Detaylandırılması

Model girdisi olarak iki farklı bilgi kanalının (modalitenin) **örnek ID'si (`sample_id`) üzerinden bire bir eşleştirilmesi** hedeflenmektedir:

1. **Görsel Modalite (RGB Trafik İşareti Görüntüsü):**
   - Gerçek yol koşullarından alınmış, farklı aydınlatma, perspektif, arka plan karmaşası ve çözünürlüklere sahip trafik işareti kırpımlarıdır (`RGB`, ön işleme sonrası `64x64` piksel).
   - Görsel şekil (sekizgen, üçgen, daire), renk (kırmızı çerçeve, mavi zemin) ve sembol/rakam örüntülerini taşır.

2. **Metinsel Modalite (Türkçe ADAS Sürücü Uyarı Metni ve Bağlamı):**
   - Her görsel örnekle `sample_id` üzerinden eşleşen; **ADAS sesli/ekran uyarı cümlesi**, **ilgili trafik kuralı tanımı** ve **görsel koşul bağlamını** (çözünürlük, parlaklık/ışık durumu) bir arada içeren çeşitlendirilmiş Türkçe metin verisidir.
   - İlerleyen final aşamasında metin kodlayıcı (ör. TF-IDF / Embedding / Transformer) ile vektörleştirilip CNN tabanlı görsel özelliklerle ortak bir gizli uzayda (feature fusion) birleştirilecektir.

---

### 1.3 Multimodal Yaklaşımın Gerekçelendirilmesi
Tek modaliteli sistemler gerçek dünya sürüş koşullarında önemli kısıtlara sahiptir:
- **Yalnızca Görüntü Kullanmanın Kısıtları:** Düşük çözünürlük, gece karanlığı, hareket bulanıklığı veya kısmi tıkanma (ağaç dalı/kir) durumlarında görsel sınıflandırıcılar birbirine benzeyen şekilleri (ör. kırmızı daire içindeki `30` vs `50` veya ters üçgen uyarı işaretlerini) karıştırabilir.
- **Yalnızca Metin Kullanmanın Kısıtları:** Sürücü uyarı metni veya bağlam bilgisi anlamsal risk düzeyini iyi taşısa da sahadaki anlık görsel kanıtı ve tabela geometrisini içermez.
- **Multimodal Füzyonun Avantajı:** Görsel öznitelikler ile metinsel kural/bağlam bilgisinin ortak bir temsilde birleştirilmesi (feature-level fusion), bir modalitedeki gürültünün diğer modalite tarafından telafi edilmesini sağlar; böylece daha yüksek sınıflandırma doğruluğu ve sürüş güvenliği açısından daha tutarlı kararlar elde edilir.

---

## 2. Proje Klasör Yapısı

```text
cok-modlu-tabela/
├── data/
│   ├── raw/                  # İndirilen ham GTSRB verileri ve ham etiket tabloları
│   └── processed/            # Eşleştirilmiş multimodal DataFrame, ön işlenmiş görseller ve split dosyaları
├── notebooks/                # Adım adım keşifsel veri analizi (EDA) ve raporlama defterleri
├── outputs/
│   ├── figures/              # EDA sürecinde üretilen 8-10+ görsel grafik çıktısı
│   └── reports/              # İstatistiksel özet tablolar ve vize analiz raporları
├── src/
│   ├── __init__.py           # Paket dışa aktarımları
│   ├── config.py             # Dizin yolları, sabitler, 4 ana sınıf ve 16 GTSRB kataloğu
│   ├── data_loader.py        # GTSRB indirme/üretme, 1:1 modalite eşleştirme ve doğrulama
│   ├── preprocessing.py      # Temizleme, 64x64 CLAHE/normalizasyon, Türkçe metin temizliği ve öncesi-sonrası
│   ├── eda.py                # İstatistiksel analizler ve 7 keşifsel görselleştirme grafiği (fig03-fig09)
│   └── splitter.py           # Sızıntısız Stratified 70/15/15 + 5-Fold bölme ve model tensör hazırlığı
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. Vize Aşaması Çalışma Planı

- [x] **Adım 1:** Proje klasör yapısının oluşturulması, problem tanımı ve multimodal sınıf/katalog yapılandırması ([src/config.py](file:///c:/Users/ferhat/Desktop/github_projeler/cok-modlu-tabela/src/config.py))
- [x] **Adım 2:** Veri setini indirme/yükleme, görsel + metin modalitelerini ortak `sample_id` anahtarı ile Pandas DataFrame'de eşleştirme ve bütünlük doğrulama modülü ([src/data_loader.py](file:///c:/Users/ferhat/Desktop/github_projeler/cok-modlu-tabela/src/data_loader.py))
- [x] **Adım 3:** Veri temizleme, görsel boyutlandırma/CLAHE/normalizasyon, Türkçe metin temizleme/tokenizasyon hazırlığı ve öncesi–sonrası karşılaştırma modülü ([src/preprocessing.py](file:///c:/Users/ferhat/Desktop/github_projeler/cok-modlu-tabela/src/preprocessing.py))
- [x] **Adım 4:** Keşifsel Veri Analizi (EDA) ile toplam 10 özgün grafik + bilimsel yorumlar ([src/eda.py](file:///c:/Users/ferhat/Desktop/github_projeler/cok-modlu-tabela/src/eda.py)) ve sızıntısız Stratified Train/Val/Test + 5-Fold bölme / tensör hazırlığı ([src/splitter.py](file:///c:/Users/ferhat/Desktop/github_projeler/cok-modlu-tabela/src/splitter.py))
- [ ] **Adım 5:** Tüm adımları birleştiren Jupyter Notebook ve Vize Analiz Raporu
