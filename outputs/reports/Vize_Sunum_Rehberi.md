# Vize Sunum ve Soru–Cevap Rehberi (5 Dakikalık Akış)

Bu rehber, **Aşama 1 (Vize)** sunumunda hocanızın karşısında ekranınızı paylaşırken hangi dosyayı sırayla açacağınızı, her adımda ne söyleyeceğinizi ve gelebilecek teknik sorulara vereceğiniz net yanıtları içerir.

> **Önemli Not:** Vize yönergesi gereği bu aşamada **model eğitimi veya canlı yapay zeka tahmini yapılmaz**. Amacımız; iki modaliteyi doğru eşleştirdiğimizi, veriyi bilimsel yöntemlerle temizleyip ön işlediğimizi, 10 farklı grafikle analiz ettiğimizi ve sızıntısız böldüğümüzü kanıtlamaktır.

---

## 1. Sunum Sırasında Açılacak Dosyalar
1. **Ana Gösterim Ekranı:** `outputs/vize_sunum_paneli.html` (Tarayıcıda çift tıklayıp açın — 5 sekmeli canlı sunum ekranıdır).
2. **Kod / Defter Kanıtı (Hoca kod isterse):** `notebooks/01_multimodal_vize_analiz.ipynb` ve `src/` altındaki modüller (`config.py`, `data_loader.py`, `preprocessing.py`, `eda.py`, `splitter.py`).
3. **Teslim Raporu:** `outputs/reports/Vize_Teknik_Raporu.md`.

---

## 2. Dakika Dakika Konuşma Akışı (Toplam ~5 Dakika)

### Dakika 1 — Problem Tanımı ve Multimodal Yapı (*Sekme 1: Kriter 1–3*)
- **Ne Açık?** `vize_sunum_paneli.html` -> **1. Sekme (Problem & Dağılım)**
- **Ne Söyleyeceksiniz?**
  > *"Hocam, projemizde otonom sürüş ve sürücü destek sistemleri (ADAS) bağlamında trafik işaretlerinin sürüş güvenliği ve kural ihlali riskini sınıflandıran iki kollu (multimodal) bir yapı tasarladık. GTSRB şemasından seçtiğimiz **16 alt trafik işaretini**, ihlal riskine göre **4 ana sınıfta** (`Kritik_Yasak_ve_Hiz`, `Oncelik_ve_Durdurma`, `Tehlike_ve_Uyari`, `Mecburi_Yon_ve_Bilgi`) topladık.*
  > *1. modalitemiz RGB trafik tabelası görselleri; 2. modalitemiz ise aynı olaya ait Türkçe ADAS sürücü uyarı metni, kural tanımı ve sensör koşul bilgisidir. Uzaktan çekilen (`20-31px`) veya gece karanlığındaki görüntülerde tek başına görsel model rakamları/şekilleri karıştırabileceği için metinsel kural bağlamını ortak uzayda birleştirmeyi hedefliyoruz."*
- **Göstereceğiniz Kanıt:** Ekrandaki **Şekil 3** — 4 ana sınıfta `548–968` örnek, 16 alt sınıfta `159–222` örnek olduğunu ve dersin *"en az 3 sınıf, sınıf başına >=150 örnek, toplam >=500 örnek"* şartının sağlandığını gösterin.

### Dakika 2 — Veri Toplama ve 1:1 Modalite Eşleştirme (*Sekme 2: Kriter 4*)
- **Ne Açık?** **2. Sekme (Etkileşimli 1:1 Örnek İnceleyici)**
- **Ne Söyleyeceksiniz?**
  > *"İki modaliteyi rastgele birleştirmedik; tekil **`sample_id` (`TRSIGN_00001`..)** anahtarı üzerinden tekilleştirip bire bir (`one_to_one`) eşleştirdik. Ham görsel tablomuzda `2.993`, ham metin tablomuzda `2.985` kayıt vardı. Önce `18` görsel ve `14` metin tekrarını ayıkladık. Ardından tek modalitesi eksik olan `26` kaydı (`15` metni olmayan görsel, `11` görseli olmayan metin) ayırarak **2.960 eşleşmiş çifte (`%99.13` eşleşme oranı)** ulaştık."*
- **Göstereceğiniz Kanıt:** Paneldeki filtrelerden bir sınıfa tıklayıp **Ham Görsel + 64×64 CLAHE Görsel + Ham/Temiz Metin** çiftini canlı gösterin ve **Şekil 6**'yı vurgulayın.

### Dakika 3 — Veri Temizleme ve Ön İşleme Öncesi–Sonrası (*Sekme 3: Kriter 5*)
- **Ne Açık?** **3. Sekme (Temizleme & Ön İşleme)**
- **Ne Söyleyeceksiniz?**
  > *"Kalite denetiminde `12` bozuk PNG dosyası, `16` adet `<20px` okunamaz görüntü ve `10` boş metin tespit edip eledik; böylece **`2.922` temiz örneğe** indik. Ayrıca `10` ham sınıf etiketi hatasını `SIGN_CATALOG` referansıyla düzelttik. Ancak gece karanlığı veya `20–31px` çözünürlükteki **`348` zorlu örneği** silmedik, bayraklayarak koruduk; çünkü gerçek trafikte araç bu koşullarla karşılaşır.*
  > *Ön işlemede görüntüleri `64x64 Bicubic` boyutuna getirip yalnızca LAB uzayının `L` kanalına **CLAHE** uyguladık (renk tonlarını bozmadan kontrastı artırmak için) ve `[0, 1]` aralığına normalize ettik. Metinde ise CAN-BUS/sensör log etiketlerini (`%41.51 -> %0.0`) temizledik, Türkçe duyarlı küçük harfe (`İ->i`, `I->ı`) çevirdik ve `MAX_SEQ_LEN=40` seçerek **%0.0 kesilme bilgi kaybıyla** `537` tokenlık sözlük oluşturduk."*
- **Göstereceğiniz Kanıt:** **Şekil 1** (Gece/düşük çözünürlük öncesi–sonrası histogramları) ve **Şekil 2** (Temizleme şelalesi & metin örneği).

### Dakika 4 — Keşifsel Veri Analizi (EDA) ve Multimodal Kanıt (*Sekme 4: Kriter 6*)
- **Ne Açık?** **4. Sekme (10 EDA Grafiği Galerisi)**
- **Ne Söyleyeceksiniz:**
  - **Şekil 4 & Şekil 5:** *"Görsellerdeki aydınlatma koşullarının sınıflar arasında yanlılık yaratmadığını ve metinde `hız`, `yasaktır`, `dur`, `kavşak`, `yaya` gibi kelimelerin güçlü sınıf imzası taşıdığını gördük."*
  - **Şekil 7 & Şekil 8:** *"Görsel metriklerle metin uzunluğu arasındaki çapraz korelasyon en fazla `|r|=0.18`; yani iki modalite birbirinin kopyası değil. 2B uzay izdüşümünde de yalnızca görselin Silhouette skoru `0.197` iken metin (`0.237`) ve füzyon (`0.211`) sınıfları daha net ayırıyor."*
  - **Şekil 9:** *"Aykırı durum analizinde koruduğumuz `348` zorlu örnekle elediğimiz `38` bozuk/okunamaz kaydı karşılaştırdık."*

### Dakika 5 — Veri Bölme Stratejisi, Final Planı ve Kod Gösterimi (*Sekme 5: Kriter 7–8*)
- **Ne Açık?** **5. Sekme (Veri Bölme & Soru-Cevap)** + gerekirse `01_multimodal_vize_analiz.ipynb`
- **Ne Söyleyeceksiniz?**
  > *"Veriyi `gtsrb_id` üzerinden katmanlı (Stratified) olarak **%70 Train (`2.045`), %15 Val (`438`), %15 Test (`439`)** diye böldük ve Train içinde **Stratified 5-Fold CV (`409`/fold)** kurduk. Bölmeler arasında ortak `sample_id` sayısı `0`'dır; görsel ve metin aynı `sample_id` satırında kilitli olduğu için veri sızıntısı imkânsızdır. Tüm tensörleri `model_ready_inputs.npz` dosyasına hazırladık. Final aşamasında bu veri üzerinde tek modaliteli CNN/BiLSTM modelleriyle Multimodal Füzyon modelini eğitip kıyaslayacağız."*

---

## 3. Hocanın Sorabileceği 6 Kritik Soru ve Hazır Yanıtları

1. **"Neden RGB görüntülere doğrudan histogram eşitleme değil de LAB-CLAHE uyguladınız?"**
   - *Yanıt:* Trafik işaretlerinde kırmızı, sarı ve mavi renkler doğrudan sınıf anlamı taşır. RGB kanallarına ayrı ayrı eşitleme yapmak renk dengesini bozar; LAB uzayının yalnızca `L` (parlaklık) kanalına CLAHE uygulamak rengi korurken karanlık veya parlamış tabelalardaki rakam/sembol kontrastını artırır.

2. **"Metin temizlemede `30`, `50`, `70` gibi sayıları veya `dur` kelimesini neden silmediniz?"**
   - *Yanıt:* Sayıları silmek `Hiz_Siniri_30` ile `Hiz_Siniri_70`'i anlamsal olarak farksız hale getirirdi. Stop-word listemizi yalnızca nötr bağlaçlarla (`ve`, `ile`, `bu`, `için`) sınırlı tuttuk; `yasaktır`, `dur`, `değil` ve hız limitlerini koruduk.

3. **"Veri sızıntısını (data leakage) nasıl önlediniz? Görsel ve metin farklı bölmelere düşebilir mi?"**
   - *Yanıt:* Düşemez. Ham aşamadaki `18` görsel ve `14` metin tekrarını `sample_id` bazında tekilleştirdik. Ardından `validate="one_to_one"` ile birleştirdiğimiz tekil tablo üzerinden `stratify=gtsrb_id` bölme yaptığımız için aynı `sample_id`'ye ait görsel ve metin daima aynı bölmede kalır; `train ∩ val ∩ test = 0` kontrolü kodla doğrulandı (`step4_split_and_model_prep_report.json`).

4. **"Eksik modaliteli 26 kaydı neden sentetik olarak doldurup kurtarmadınız?"**
   - *Yanıt:* Eksik modaliteli kayıtlar tüm adayların yalnızca `%0.87`'sini (`26 / 2.986`) oluşturuyor. Bu kadar küçük bir pay için yapay görüntü/metin üretmek, modele gerçek dışı veya ezbere dayalı sahte eşleşmeler öğretme riski taşıdığından inner join ile kapsam dışı bırakmak daha sağlıklıdır.

5. **"348 aykırı örneği neden silmediniz?"**
   - *Yanıt:* Bu `348` kayıt bozuk değil; gece karanlığı (`parlaklık < 75`), uzaktan çekim (`20–31px`) veya güneş parlaması (`> 165`) içeren geçerli fakat zorlu sürüş koşullarıdır. Bunları silersek model yalnızca ideal gündüz fotoğraflarına ezberler (overfitting); ancak `<20px` okunamayan `16` örneği ve `12` bozuk PNG'yi temizlikte eledik.

6. **"Neden Stratified 5-Fold kullandınız da Group-based veya Temporal bölme kullanmadınız?"**
   - *Yanıt:* Veri setindeki her örnek bağımsız bir trafik tabelası tespitidir; ardışık kare tekrarları (`18` adet) Adım 2'de zaten temizlenmiştir. Zamana veya sürücü/hasta kimliğine bağlı bir kümeleme olmadığı için hem 4 ana sınıfın hem 16 alt sınıfın oranlarını koruyan Stratified bölme en uygun yöntemdir.

