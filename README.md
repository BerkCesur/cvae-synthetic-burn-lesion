# cVAE Tabanlı Üretken Model ile Sentetik Yanık Lezyonu Üretimi

[cite_start]Tıbbi yapay zeka alanında sıkça karşılaşılan veri kıtlığı ve sınıf dengesizliği (Data Imbalance) problemlerine çözüm sunmak amacıyla geliştirilmiş, yanık derecesi koşullu sentetik veri üretim modelidir[cite: 59, 65]. [cite_start]Model, ağır yanık vakalarındaki az örnek sayısını artırarak tanı başarısını desteklemeyi [cite: 60] [cite_start]ve KVKK/GDPR kısıtlamalarına takılmadan sağlık verisi paylaşımını mümkün kılmayı hedefler[cite: 62, 63].

## Proje Özeti ve Teknik Yaklaşımlar
* [cite_start]**Veri Seti:** Kaggle Skin Burn Dataset kullanılmıştır[cite: 68]. [cite_start]YOLO bbox etiketli 1225 ham görüntüden, koordinatların piksel normalize edilmesi ve yama çıkarma (patch extraction) yöntemiyle 2213 adet eğitim örneği elde edilmiştir[cite: 10, 68]. 
* [cite_start]**Sınıf Dağılımı:** Veri setinde 1. Derece (872), 2. Derece (987) ve 3. Derece (354) yanık örnekleri bulunmaktadır[cite: 8, 9].
* [cite_start]**Ön İşleme & KVKK Uyumu:** OpenCV Haar Cascade kullanılarak yüz tespiti yapılmış ve arka plan rengiyle maskelenerek hasta gizliliği (KVKK/GDPR uyumu) sağlanmıştır[cite: 9, 10]. [cite_start]Bu sayede modelin odağı gereksiz yüz verilerinden kurtarılarak doğrudan lezyona çekilmiştir[cite: 10].
* [cite_start]**Model Mimarisi:** Conditional Variational Autoencoder (cVAE) mimarisi kullanılarak belirli bir derece koşuluna bağlı üretim gerçekleştirilmiştir[cite: 65].
* **Eğitim Stabilitesi:** Üretici modellerin eğitim sürecinde sıkça karşılaşılan "posterior collapse" sorunu, KL Annealing tekniği ile başarılı bir şekilde önlenmiştir.
* **Kayıp Fonksiyonu (Loss) Optimizasyonu:** Standart MSE (Ortalama Kare Hatası) kullanımının yol açtığı görüntü bulanıklığı problemini çözmek için modele VGG-16 Perceptual Loss (algısal kayıp) entegre edilmiştir.

## Deneysel Sonuçlar ve Başarım
* **FID Skoru İyileşmesi:** Modele entegre edilen VGG-16 Perceptual Loss optimizasyonu sayesinde, üretilen sentetik görüntülerin gerçekçiliğini ölçen FID (Fréchet Inception Distance) skoru MSE'ye kıyasla %26.4 oranında iyileşerek 336.86'dan 247.78'e düşürülmüştür. Renk ve doku geçişleri çok daha belirgin hale getirilmiştir.
* **Latent Uzay Analizi:** t-SNE analizleri sonucunda modelin latent uzayında biyolojik açıdan tutarlı bir dağılım elde edildiği kanıtlanmıştır.

## Kullanılan Teknolojiler
* Python, PyTorch, OpenCV, YOLO (Anotasyon İşleme), VGG-16 (Perceptual Loss Mimarisi)
