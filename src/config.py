"""Proje genelinde kullanılan dizin yolları, sabitler ve multimodal sınıf tanımları.

Bu modül; GTSRB trafik işareti alt sınıflarını 4 ana sürüş güvenliği ve kural
ihlali kategorisine eşler, vize aşamasındaki veri toplama, temizleme, EDA ve
veri bölme adımları için tek doğruluk kaynağı (single source of truth) sağlar.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

# ------------------------------------------------------------------------------
# 1. Dizin Yolları
# ------------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

OUTPUTS_DIR: Path = BASE_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# ------------------------------------------------------------------------------
# 2. Tekrarlanabilirlik, Ön İşleme ve Vize Kriteri Sabitleri
# ------------------------------------------------------------------------------
RANDOM_SEED: int = 42

# GTSRB görselleri değişken boyutlu olduğundan tüm görseller ortak giriş
# boyutuna normalize edilir; (64, 64) ince detayları korurken bellek maliyetini dengeler.
IMAGE_SIZE: Tuple[int, int] = (64, 64)
IMAGE_CHANNELS: int = 3

# Ders kapsamında belirlenen minimum veri yeterlilik kuralları
MIN_SAMPLES_PER_CLASS: int = 150
MIN_TOTAL_SAMPLES: int = 500

# Veri sızıntısını önlemek için katmanlı (stratified) bölme oranları
SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# ------------------------------------------------------------------------------
# 3. Ana Sürüş Güvenliği ve Kural İhlali Sınıfları (4 Sınıf)
# ------------------------------------------------------------------------------
SAFETY_CATEGORIES: Dict[int, Dict[str, str]] = {
    0: {
        "code": "Kritik_Yasak_ve_Hiz",
        "label_tr": "Kritik Yasak ve Hız Sınırları",
        "risk_level": "Çok Yüksek",
        "description": (
            "İhlal edilmesi doğrudan ağır trafik cezası veya yüksek enerjili "
            "çarpışma riski doğuran hız limiti, geçme yasağı ve giriş yasağı işaretleri."
        ),
    },
    1: {
        "code": "Oncelik_ve_Durdurma",
        "label_tr": "Öncelik ve Durdurma İşaretleri",
        "risk_level": "Çok Yüksek",
        "description": (
            "Kavşak ve kesişme noktalarında geçiş hakkını belirleyen, uyulmadığında "
            "yandan çarpışma (T-bone) kazalarına yol açan kritik kontrol işaretleri."
        ),
    },
    2: {
        "code": "Tehlike_ve_Uyari",
        "label_tr": "Tehlike ve Önleyici Uyarılar",
        "risk_level": "Orta-Yüksek",
        "description": (
            "Yol çalışması, yaya geçidi veya kaygan zemin gibi çevresel tehlikeleri "
            "bildirerek sürücünün erken hız düşürmesini gerektiren uyarı işaretleri."
        ),
    },
    3: {
        "code": "Mecburi_Yon_ve_Bilgi",
        "label_tr": "Mecburi Yön ve Trafik Akışı",
        "risk_level": "Düşük-Orta",
        "description": (
            "Araçların şerit ve dönüş yönünü düzenleyerek ters yön ihlallerini ve "
            "kavşak içi akış karmaşasını önleyen yönlendirme işaretleri."
        ),
    },
}

# ------------------------------------------------------------------------------
# 4. Seçili GTSRB Alt Sınıfları ve Multimodal Metin/Kural Kataloğu (16 Alt Sınıf)
# ------------------------------------------------------------------------------
# Her sınıfın GTSRB ID'si korunarak ana güvenlik sınıfına ve zengin Türkçe ADAS
# sürücü uyarı şablonlarına eşlenmesi sağlanır.
SIGN_CATALOG: Dict[int, Dict[str, Any]] = {
    # --- Sınıf 0: Kritik_Yasak_ve_Hiz ---
    1: {
        "gtsrb_id": 1,
        "code_name": "Hiz_Siniri_30",
        "turkish_name": "Azami Hız Sınırı 30 km/sa",
        "safety_class_id": 0,
        "safety_class_name": "Kritik_Yasak_ve_Hiz",
        "risk_level": "Çok Yüksek",
        "rule_definition": "Araç hızı en fazla 30 km/sa olmalıdır; yerleşim veya okul bölgesi sınırıdır.",
        "adas_templates": [
            "Dikkat: Azami hız sınırı 30 km/sa bölgesine girildi, hızınızı derhal düşürün.",
            "ADAS Uyarı: İleride 30 km/sa hız limiti levhası algılandı, kural ihlali riskini önlemek için yavaşlayın.",
            "Güvenlik Bildirimi: Düşük hız bölgesi (30 km/sa). Yaya ve okul çevresi hassasiyeti nedeniyle hızınızı sınırlayın.",
        ],
    },
    2: {
        "gtsrb_id": 2,
        "code_name": "Hiz_Siniri_50",
        "turkish_name": "Azami Hız Sınırı 50 km/sa",
        "safety_class_id": 0,
        "safety_class_name": "Kritik_Yasak_ve_Hiz",
        "risk_level": "Yüksek",
        "rule_definition": "Şehir içi ana arter standardı olarak azami hız 50 km/sa ile sınırlandırılmıştır.",
        "adas_templates": [
            "Uyarı: Azami hız sınırı 50 km/sa. Şehir içi hız limitine uyum sağlayın.",
            "ADAS Bildirimi: 50 km/sa hız tahdidi tespit edildi, gaz pedalından ayağınızı çekerek hızı dengeleyin.",
            "Trafik Kuralı: Bu kesimde 50 km/sa üzerindeki hızlar kural ihlali ve takip mesafesi riski oluşturur.",
        ],
    },
    4: {
        "gtsrb_id": 4,
        "code_name": "Hiz_Siniri_70",
        "turkish_name": "Azami Hız Sınırı 70 km/sa",
        "safety_class_id": 0,
        "safety_class_name": "Kritik_Yasak_ve_Hiz",
        "risk_level": "Yüksek",
        "rule_definition": "Bölünmüş yol veya çevre yolu yaklaşımında azami hız 70 km/sa ile kısıtlanmıştır.",
        "adas_templates": [
            "Dikkat: Azami hız sınırı 70 km/sa levhası algılandı, seyir hızınızı kontrol edin.",
            "ADAS Uyarı: 70 km/sa hız limiti başladı; ani hızlanmalardan kaçınarak şerit takibini koruyun.",
            "Kural Hatırlatması: Bu yol kesiminde yasal hız üst sınırı 70 km/sa olarak belirlenmiştir.",
        ],
    },
    9: {
        "gtsrb_id": 9,
        "code_name": "Gecme_Yasagi",
        "turkish_name": "Öndeki Aracı Geçmek Yasaktır",
        "safety_class_id": 0,
        "safety_class_name": "Kritik_Yasak_ve_Hiz",
        "risk_level": "Çok Yüksek",
        "rule_definition": "Karşı şeride geçerek sollama yapmak kesinlikle yasaktır.",
        "adas_templates": [
            "Kritik Uyarı: Sollama yasağı başladı, öndeki aracı geçmeye teşebbüs etmeyin.",
            "ADAS Uyarı: Geçme yasağı tabelası algılandı; şeridinizi koruyun ve takip mesafesini artırın.",
            "Güvenlik Uyarısı: Görüş kısıtı veya karşı trafik riski nedeniyle sollama yasaklanmıştır.",
        ],
    },
    17: {
        "gtsrb_id": 17,
        "code_name": "Girisi_Olmayan_Yol",
        "turkish_name": "Girişi Olmayan Yol (Tek Yön İhlali)",
        "safety_class_id": 0,
        "safety_class_name": "Kritik_Yasak_ve_Hiz",
        "risk_level": "Çok Yüksek",
        "rule_definition": "Bu yönden taşıt girişi yasaktır; ilerlemek ters yön ihlaline yol açar.",
        "adas_templates": [
            "ACİL UYARI: Girişi olmayan yol levhası tespit edildi! Ters yöne girmeyin ve durun.",
            "ADAS Kritik Alarm: Taşıt trafiğine kapalı giriş yönü algılandı, alternatif güzergaha yönelin.",
            "Kural İhlali Uyarısı: İlerideki yol kesimi karşı yöne tek yönlüdür, giriş kesinlikle yasaktır.",
        ],
    },
    # --- Sınıf 1: Oncelik_ve_Durdurma ---
    12: {
        "gtsrb_id": 12,
        "code_name": "Oncelikli_Yol",
        "turkish_name": "Ana Yol / Öncelikli Yol",
        "safety_class_id": 1,
        "safety_class_name": "Oncelik_ve_Durdurma",
        "risk_level": "Yüksek",
        "rule_definition": "Bulunduğunuz yol kavşaklarda geçiş önceliğine sahiptir; yine de tali yol çıkışları izlenmelidir.",
        "adas_templates": [
            "Bilgi: Öncelikli yol (ana yol) üzerindesiniz, kavşaklarda tali yoldan çıkabilecek araçlara dikkat edin.",
            "ADAS Bildirimi: Geçiş üstünlüğü bu şerittedir; sabit hızla ilerlerken kavşak taramasını sürdürün.",
            "Kavşak Uyarısı: Ana yol tabelası algılandı, geçiş hakkı sizde olsa da savunmacı sürüşü bırakmayın.",
        ],
    },
    13: {
        "gtsrb_id": 13,
        "code_name": "Yol_Ver",
        "turkish_name": "Yol Ver",
        "safety_class_id": 1,
        "safety_class_name": "Oncelik_ve_Durdurma",
        "risk_level": "Çok Yüksek",
        "rule_definition": "Kavşaktaki veya ana yoldaki araçlara geçiş hakkı tanınmalı, gerekirse durulmalıdır.",
        "adas_templates": [
            "Dikkat: Yol ver levhası algılandı, ana yoldaki trafiğe geçiş hakkı tanımak için yavaşlayın.",
            "ADAS Uyarı: İlerideki kavşakta öncelik karşı yöndedir, fren hazırlığı yapın ve yolu kontrol edin.",
            "Kavşak Güvenliği: Geçiş önceliğine uymamak yandan çarpışma riskini artırır, gerektiğinde durun.",
        ],
    },
    14: {
        "gtsrb_id": 14,
        "code_name": "Dur",
        "turkish_name": "DUR (Tam Durma Zorunluluğu)",
        "safety_class_id": 1,
        "safety_class_name": "Oncelik_ve_Durdurma",
        "risk_level": "Çok Yüksek",
        "rule_definition": "Dur çizgisinde tekerlekler tamamen durmalı ve tüm yönler kontrol edildikten sonra hareket edilmelidir.",
        "adas_templates": [
            "KRİTİK UYARI: DUR levhası algılandı! Kavşak girişinde aracınızı tamamen durdurun.",
            "ADAS Acil Bildirim: Tam durma zorunluluğu var; hızınızı sıfıra indirin ve çapraz trafiği kontrol edin.",
            "Yüksek İhlal Riski: DUR işaretinde duraksamadan geçmek ağır kural ihlali ve çarpışma nedenidir.",
        ],
    },
    # --- Sınıf 2: Tehlike_ve_Uyari ---
    18: {
        "gtsrb_id": 18,
        "code_name": "Dikkat_Tehlike",
        "turkish_name": "Dikkat (Genel Tehlike Uyarısı)",
        "safety_class_id": 2,
        "safety_class_name": "Tehlike_ve_Uyari",
        "risk_level": "Orta-Yüksek",
        "rule_definition": "İleride öngörülemeyen yol veya trafik tehlikesi bulunmaktadır; hız azaltılmalıdır.",
        "adas_templates": [
            "Dikkat: Genel tehlike uyarı levhası algılandı, hızınızı düşürerek çevreyi gözlemleyin.",
            "ADAS Uyarı: İleride beklenmedik yol koşulu olabilir, takip mesafenizi artırın.",
            "Önleyici Uyarı: Yol üzerindeki olağan dışı duruma karşı frenlemeye hazır şekilde ilerleyin.",
        ],
    },
    25: {
        "gtsrb_id": 25,
        "code_name": "Yol_Calismasi",
        "turkish_name": "Yol Çalışması",
        "safety_class_id": 2,
        "safety_class_name": "Tehlike_ve_Uyari",
        "risk_level": "Orta-Yüksek",
        "rule_definition": "Bakım/onarım nedeniyle şerit daralması, iş makineleri ve yol işçileri bulunabilir.",
        "adas_templates": [
            "Uyarı: İleride yol çalışması var! Şerit daralmasına ve saha görevlilerine karşı yavaşlayın.",
            "ADAS Bildirimi: Yol bakım bölgesi algılandı, geçici trafik işaretçilerini takip edin.",
            "Güvenlik Uyarısı: Çalışma alanında ani şerit değişimi ve zemin bozulması riskine karşı dikkatli olun.",
        ],
    },
    27: {
        "gtsrb_id": 27,
        "code_name": "Yaya_Gecidi",
        "turkish_name": "Yaya Geçidi Yaklaşımı",
        "safety_class_id": 2,
        "safety_class_name": "Tehlike_ve_Uyari",
        "risk_level": "Çok Yüksek",
        "rule_definition": "Yaya geçidine yaklaşırken hız düşürülmeli, geçmek üzere olan yayalara ilk geçiş hakkı verilmelidir.",
        "adas_templates": [
            "Dikkat: Yaya geçidi algılandı! Yayalara öncelik vermek için hızınızı azaltın.",
            "ADAS Yaya Uyarısı: İleride yaya geçidi bulunuyor, kaldırım kenarındaki hareketliliği izleyin.",
            "Sürüş Güvenliği: Yaya geçidinde durmamak ciddi kural ihlalidir, fren mesafenizi ayarlayın.",
        ],
    },
    30: {
        "gtsrb_id": 30,
        "code_name": "Buzlanma_Uyarisi",
        "turkish_name": "Gizli Buzlanma ve Kar Tehlikesi",
        "safety_class_id": 2,
        "safety_class_name": "Tehlike_ve_Uyari",
        "risk_level": "Orta-Yüksek",
        "rule_definition": "Düşük sıcaklık nedeniyle yol yüzeyi kayganlaşabilir; ani fren ve sert direksiyon hareketinden kaçınılmalıdır.",
        "adas_templates": [
            "Zemin Uyarısı: Gizli buzlanma riski algılandı, hızınızı kademeli düşürün ve ani fren yapmayın.",
            "ADAS Bildirimi: Kaygan zemin / kar tehlikesi nedeniyle takip mesafenizi iki katına çıkarın.",
            "Güvenlik Hatırlatması: Düşük tutuş koşullarında savrulmayı önlemek için yumuşak manevra yapın.",
        ],
    },
    # --- Sınıf 3: Mecburi_Yon_ve_Bilgi ---
    33: {
        "gtsrb_id": 33,
        "code_name": "Saga_Mecburi_Yon",
        "turkish_name": "İleriden Sağa Mecburi Yön",
        "safety_class_id": 3,
        "safety_class_name": "Mecburi_Yon_ve_Bilgi",
        "risk_level": "Düşük-Orta",
        "rule_definition": "Kavşak veya yol ayrımında yalnızca sağa dönüş yapılması zorunludur.",
        "adas_templates": [
            "Yönlendirme: İleriden sağa mecburi yön levhası algılandı, sağ sinyalinizi vererek sağ şeride geçin.",
            "ADAS Bilgi: Bu noktada düz veya sola devam etmek yasaktır; sağa dönüşe hazırlanın.",
            "Akış Uyarısı: Trafik akışı yalnızca sağ yöne zorunludur, dönüş hızınızı ayarlayın.",
        ],
    },
    34: {
        "gtsrb_id": 34,
        "code_name": "Sola_Mecburi_Yon",
        "turkish_name": "İleriden Sola Mecburi Yön",
        "safety_class_id": 3,
        "safety_class_name": "Mecburi_Yon_ve_Bilgi",
        "risk_level": "Düşük-Orta",
        "rule_definition": "Kavşak veya yol ayrımında yalnızca sola dönüş yapılması zorunludur.",
        "adas_templates": [
            "Yönlendirme: İleriden sola mecburi yön algılandı, sol sinyali verip uygun şeride konumlanın.",
            "ADAS Bilgi: Sağ veya düz istikamet kapalıdır; yalnızca sola dönüş yapabilirsiniz.",
            "Akış Uyarısı: Sola mecburi dönüş öncesi karşı şeridi ve ayna kör noktalarını kontrol edin.",
        ],
    },
    35: {
        "gtsrb_id": 35,
        "code_name": "Ileri_Mecburi_Yon",
        "turkish_name": "İleri Mecburi Yön",
        "safety_class_id": 3,
        "safety_class_name": "Mecburi_Yon_ve_Bilgi",
        "risk_level": "Düşük-Orta",
        "rule_definition": "Araçların sağa veya sola sapmadan yalnızca ileri istikamette devam etmesi zorunludur.",
        "adas_templates": [
            "Yönlendirme: İleri mecburi yön levhası algılandı, şeridinizi koruyarak düz devam edin.",
            "ADAS Bilgi: Bu kavşakta sağa ve sola dönüşler yasaklanmıştır, ileri yönde ilerleyin.",
            "Akış Uyarısı: Yön ihlalini önlemek için dönüş manevralarından kaçının ve akışı takip edin.",
        ],
    },
    38: {
        "gtsrb_id": 38,
        "code_name": "Sagdan_Gidiniz",
        "turkish_name": "Sağdan Gidiniz (Refüj / Engel Ayrımı)",
        "safety_class_id": 3,
        "safety_class_name": "Mecburi_Yon_ve_Bilgi",
        "risk_level": "Orta",
        "rule_definition": "Yol ayrımındaki ada veya engelin sağ tarafından geçilmesi zorunludur.",
        "adas_templates": [
            "Yönlendirme: Sağdan gidiniz levhası tespit edildi, trafik adasının sağ şeridinden devam edin.",
            "ADAS Bilgi: Orta refüj ayrımı yaklaşıyor; ters bölmeye girmemek için sağa yönelin.",
            "Şerit Güvenliği: Engelin sağından geçerek karşı yön trafiğiyle çakışma riskini önleyin.",
        ],
    },
}


def ensure_directories() -> None:
    """Projenin veri ve çıktı klasörlerinin disk üzerinde var olmasını sağlar."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

