# cVAE Tabanlı Üretken Model ile Sentetik Yanık Lezyonu Üretimi

Bu proje, tıbbi görüntüleme ve sağlık teknolojileri (Medical AI) alanında sıkça karşılaşılan veri kıtlığı (Data Scarcity) ve sınıf dengesizliği (Class Imbalance) problemlerine analitik çözümler sunmak amacıyla tasarlanmış, yanık derecesi koşullu bir sentetik veri üretim modelidir.

Ağır yanık vakalarına ait klinik verilerin azlığı, derin öğrenme modellerinin tanı ve sınıflandırma başarısını doğrudan düşürmektedir. Ayrıca KVKK ve GDPR gibi hasta gizliliği mevzuatları, gerçek tıbbi verilerin kurumlar arası paylaşımını ve açık kaynaklı kullanımını kısıtlamaktadır. Bu çalışma, hem azınlık sınıflarını veri artırma (data augmentation) yöntemleriyle desteklemeyi hem de hasta mahremiyetini ihlal etmeden yüksek gerçekçilikte sentetik lezyon verisi üretmeyi başarmıştır.

---

## Proje Mimarisi ve Teknik Yaklaşımlar

### 1. Veri İşleme Hattı (Data Pipeline)
* **Veri Kaynağı:** Kaggle Skin Burn Dataset kullanılmıştır.
* **Yama Çıkarma (Patch Extraction):** Modelin arka plan gürültülerinden ve ilgisiz detaylardan arındırılarak doğrudan lezyon dokusuna odaklanması amacıyla, YOLO bounding box (sınır kutusu) koordinatları kullanılmıştır. Piksel normalizasyonu yapılan 1225 adet ham görüntüden toplam 2213 adet yüksek kaliteli eğitim yaması (patch) elde edilmiştir.
* **Sınıf Dağılım Analizi:** Veri setindeki sınıfsal dengesizlik seviyesi tespit edilmiş ve model bu dağılımsal parametrelere göre koşullandırılmıştır:
  * 1. Derece Yanık: 872 örnek
  * 2. Derece Yanık: 987 örnek
  * 3. Derece Yanık (Azınlık Sınıfı): 354 örnek
* **Anonimleştirme ve KVKK Uyumluluğu:** Görüntülerde yer alan insan yüzleri ve ayırt edici bölgeler, OpenCV (Haar Cascade) mimarisiyle tespit edilerek arka plan rengiyle maskelenmiştir. Böylece veri işleme süreçleri tamamen anonim bir yapıya kavuşturulmuştur.

### 2. Model Tasarımı (cVAE)
Çalışmada, rastgele ve kontrolsüz üretim yapan standart VAE veya GAN yapıları yerine Conditional Variational Autoencoder (cVAE) mimarisi tercih edilmiştir. Bu sayede model, latent uzaydan rastgele örnekleme yapmak yerine, sisteme girdi olarak verilen yanık derecesi (1, 2 veya 3) koşuluna özel ve hedeflenmiş lezyon morfolojileri üretebilmektedir.

### 3. Mühendislik Çözümleri ve Optimizasyonlar
Üretici derin öğrenme modellerinin eğitim süreçlerinde karşılaşılan iki kritik kronik problem, projeye özgü ileri düzey optimizasyon teknikleriyle çözülmüştür:

* **Posterior Collapse (Gizli Uzay Çökmesi) Çözümü:** VAE eğitimlerinde dekoderin latent uzayı görmezden gelerek girdileri doğrudan ezberleme eğilimini (posterior collapse) engellemek amacıyla KL Annealing (Kullback-Leibler Tavlama) tekniği uygulanmıştır. KL Divergence ağırlığı eğitim boyunca kademeli olarak artırılmış ve KL kaybı 0.097 seviyesinde stabilizasyona ulaştırılmıştır.
* **Bulanıklık (Blurriness) Probleminin Çözümü:** Piksel tabanlı standart MSE (Ortalama Kare Hatası) kayıp fonksiyonlarının yol açtığı ortalama odaklı bulanık görüntü üretimini engellemek için modele VGG-16 Perceptual Loss (Algısal Kayıp) mimarisi entegre edilmiştir (perc_weight = 0.1). Böylece model, pikselleri birebir kopyalamak yerine insan gözünün ve evrişimli katmanların algıladığı üst düzey öznitelikleri, renk gradyanlarını ve doku geçişlerini öğrenmiştir.

---

## Deneysel Sonuçlar ve Başarım Metrikleri

Modelin başarısı ve üretilen sentetik verilerin matematiksel kalitesi, literatürde kabul görmüş istatistiksel metriklerle doğrulanmıştır:

* **FID (Fréchet Inception Distance) Skoru İyileşmesi:**
  * Yalnızca MSE tabanlı temel modelin (Base Model) FID Skoru: 336.86
  * VGG-16 Perceptual Loss entegreli gelişmiş cVAE modelinin FID Skoru: 247.78
  * **Değerlendirme:** Geliştirilen mühendislik yaklaşımı sayesinde sentetik görüntülerin gerçekçilik ve dağılım kalitesi %26.4 oranında iyileştirilmiş; doku yapaylığı minimize edilmiştir.
* **Latent Uzay Dağılım Analizi:** t-SNE (t-Distributed Stochastic Neighbor Embedding) algoritması kullanılarak yapılan boyut azaltma analizlerinde, modelin latent uzayda 1., 2. ve 3. derece yanıkları biyolojik ve klinik gerçekliğe uygun olarak birbirinden ayrılabilir kümeler halinde organize ettiği kanıtlanmıştır.

---

## Kullanılan Teknolojiler ve Kütüphaneler
* **Programlama Dili:** Python
* **Derin Öğrenme Çatısı:** PyTorch
* **Görüntü İşleme ve Analitik Araçlar:** OpenCV, Matplotlib, Scikit-learn
* **Temel Mimariler:** cVAE, VGG-16, YOLO (Veri Etiketleme ve Bölütleme)
* **Donanım Altyapısı:** Kaggle (NVIDIA Tesla T4 GPU)

---

## Kurulum ve Çalıştırma

Projeyi yerel ortamınızda doğrulamak ve çalıştırmak için aşağıdaki adımları uygulayabilirsiniz:

1. Depoyu klonlayın:
```bash
git clone [https://github.com/berkburhancesur/cvae-synthetic-burn-lesion.git](https://github.com/berkburhancesur/cvae-synthetic-burn-lesion.git)
cd cvae-synthetic-burn-lesion
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Modeli Jupyter Notebook arayüzü üzerinden tetikleyin:
```bash
jupyter notebook "sentetik-yanik-yaralanma-verileri-VGG.ipynb"
```
