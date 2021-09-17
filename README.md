# Guvenli-Surus-Sistemi
Mezuniyet projesi olarak gerçekleştirilmiştir. Projede, araç sürücüsünün yorgunluk, güvenlik ve duygu durumu için makine öğrenmesi algoritmaları kullanılmıştır. Pyqt5 kullanılarak algoritmalar tek programda birleştirilmiştir.

Projede, yorgunluk tespiti için 2 farklı metod kullanılmıştır. Duygu durumu tanıma için de 2 farklı metod kullanılmıştır.

# Preparation of Project
Aşağıdaki link kullanılarak eğitimlerde kullanılan veri setlerine, modellere, ve dlib için gerekli olan shape predictor modeline ulaşabilirsiniz.
https://drive.google.com/drive/folders/1OkmOYf46hUi7iC9lusrLaQm5qCdmzhji?usp=sharing

Dosyada bulunan trainingSet, testSet klasörleri 1_Yorgunluk_Hog_svm 'nun içine atılmalı
Dosyada bulunan VGG16 modeli 3_Duygu_tanıma_hazır\modeller 'nun içine atılmalı
Dosyada bulunan shape_predictor, 1_Yorgunluk_dlib, 1_Yorgunluk_Hog_svm 'nun içine atılmalı



## 1_Yorgunluk_Dlib
Sistemde Opencv, Dlib, Scipy kütüphaneleri kullanılmıştır. Kodda ilk olarak OpenCV kütüphanesiyle kameradan görüntü alınmaya başlıyor. Dlib kütüphanesinden get_frontal_face_detector ve shape_predictor classlarından obje tanımlanıyor. Video görüntüleri, tek tek resim framelerinden oluştuğundan sonsuz döngü oluşturuluyor. Her bir döngüde fps oranına göre frameler tek tek alınıp kodun ileriki aşamalarında işleniyor. Alınan frameler BGR renk formatındadır. Görüntüler üzerinde işlem yapabilmek için BGR renk formatı GRAY renk formatına çevrilmiştir. Gray scale deki görüntü kullanılarak get_frontal_face_detector ile görüntüdeki yüzler tespit edilmektedir. Tespit edilen yüzler kullanılarak shape_predictor kullanılarak yüzün belirli noktaları çıkarılmaktadır. (Şekil 3-8)

![alt text](https://github.com/Berkayyldrm/Guvenli-Surus-Sistemi/blob/main/img/Resim1.png)
Şekil 3 8  Face Landmark

Yukarıda belirtilen noktalardan 36-42 arası sol gözü, 42-48 arası sağ gözü, 48-60 arası ağzı temsil etmektedir.

Bu noktalar kullanılarak sol göz, sağ göz ve ağız için her biri için ayrı döngüde bu noktaların konumları ekranda çizilerek kullanıcıya gösterilecektir. Bu noktaların x, y koordinatları daha önceden tanımlanan dizilerin içine kaydedilecektir. Gözler ve ağız için mesafe ölçme fonksiyonları tanımlanmıştır. Örnek olarak Şekil 3-9’daki sol göz için hesaplamalar aşağıdaki gibidir.

![alt text](https://github.com/Berkayyldrm/Guvenli-Surus-Sistemi/blob/main/img/Resim2.png)
Şekil 3 9 Sol Göz Landmarkı

Gözlerin açıklık oranlarının bulunması aşağıdaki formüle göre yapılmıştır.

	Göz açıklık oranı=(A+B)/(2*C)(denklem 3.1)

A:37-41 arası mesafe

B:38-40 arası mesafe

C:36-39 arası mesafe

Ağzın açıklık oranının bulunması aşağıdaki formüle göre yapılmıştır.

![alt text](https://github.com/Berkayyldrm/Guvenli-Surus-Sistemi/blob/main/img/Resim3.png)
Şekil 3 10 Ağız Landmarkı

	Ağız açıklık oranı=(A+B)/(2*C)(denklem 3.2)
	
A:50-58 arası mesafe

B:52-56 arası mesafe

C:48-54 arası mesafe

Mesafeler ölçülürken Scipy kütüphanesinin spatial fonksiyonun distance metodu kullanılmıştır. Distance metodunun Euclidean özelliği ile mesafeler ölçülmüştür.

Sağ ve sol gözün açıklık değerleri ayrı bir fonksiyonda ortalamaları alınarak kullanılmıştır. Bu ortalama değere göre gözün açıklık kapalılık durumu bir tane kesim değerine göre belirlenmiştir. Kesim değeri 0.24 olarak alınmıştır. 0.24 üzeri değerler gözün açık olduğunu, 0.24 altı değerler ise gözün kapalı olduğunu belirtmektedir. Gözün açık veya kapalı olduğu durumlar bir tane diziye kaydedilmektedir. Bu diziden de son 150 frame deki değerler alınmaktadır. Sistemin fps değeri 10 olduğundan yani son 15 saniyedeki değerler alınmaktadır.

Bu alınan değerler Perclos fonksiyonuna gönderilmektedir. Bu fonksiyonda Perclos algoritması bulunmaktadır. Perclos algoritması gözdeki yorgunluk tespitini göz durumlarına kullanarak tespit eden bir algoritmadır. Aşağıda algoritma verilmiştir.

	Toplam frame = açık gözün olduğu frame + kapalı gözün olduğu frame
	
	Perclos değeri = kapalı gözün olduğu frame * 100 / Toplam frame
	
Perclos değerinin eşik değeride 60 olarak belirlenmiştir. 60 üzeri değerde sürücü “Yorgun” olarak belirtiliyor. 60 altında değerde ise sürücü “Normal” olarak belirtiliyor. Bu elde edilen sonuç masaüstü uygulamasında gösteriliyor.

Esneme tespit fonksiyonunda ise önceden elde edilmiş mesafe değerini kullanılmıştır. Bu mesafe değerinin eşik değeri 60 olarak belirtilmiştir. 60 üzeri değerler esneme gerçekleşiyor olduğunu belirtiliyor. 60 altı değerler ise esneme olmadığını belirtmektedir. Değerin 60 üzeri olduğu durumlarda counter devreye girmektedir. Her 60 üzeri framede counter bir artmaktadır. Sistemin fps i 10 olduğundan 1 saniyelik esneme, counter’ı on arttırmaktadır. Esneme tespit fonksiyonuna 1 adet te timer eklenmiştir. Her otuz saniyede bir timer sıfırlanmaktadır ve sürücünün esneme durumuna göre yorgun veya normal olduğu masaüstü uygulamasında gösterilmektedir. Esneme durumuna göre yorgunluk tespitide counter değerinin 30 un altında olup olmadığına göre yapılmaktadır. Yani sürücü otuz saniyede üç kere esnediyse sistem sürücüye yorgunluktan dolayı uyarı vermektedir.

## 1_Yorgunluk_Hog_SVM
OpenCV, Sklearn, Histogram Of Oriented Gradients (HOG) ve aynı zamanda makine öğrenmesi destek vektör makineleri (DVM) ile yüz ve göz algılanması gerçekleştirilmiştir. Daha sonrasında PERCLOS uygulaması ile yorgunluk tespiti yapıldı.

Proje kapsamında Yönlendirilmiş Gradyanların Histogramı (HOG) ile yüz ve göz algılama özellikleri tanımlanmıştır. Daha sonrasında makine öğrenmesi algoritması olan DVM ile eğitime tabi tutulmuştur. 5 bine yakın resimle eğitime tabi tuttuk. Bu eğitim sonunda HOG ile tanımlanan yüz ve göz daha sonrasında DVM ile eğitilerek gözün kapalılık ve açıklık durumunu saptadı. Bu noktadan sonra PERCLOS uygulamasının matematiksel hesaplamasını da tamamlayarak gözün kapalılık ve açıklık durumlarında kişinin yorgunluk durumu saptandı.

## 2_Yüz_Tanıma
Bu bölüm üç ana kısımdan oluşmaktadır. Bu üç kısım, yüz fotoğraflarından oluşan veri tabanı oluşturma, veri tabanını kullanarak modeli eğitme ve eğitilen modelin kullanılmasından oluşmaktadır.
### Veri Tabanı Oluşturma 
Opencv kütüphanesi kullanılmıştır. Opencv sayesinde kameradan gelen görüntüler işlenebilmektedir. Program çalıştırıldığında kullanıcıdan ID numarası istenmektedir. ID girildikten sonra program yüz elli adet görüntü veri tabanına kaydedilmektedir. Bu kaydedilen görüntü veri tabanına, User.Id.alınankaçıncıgörüntü.jpg olarak kaydedilmektedir. Örnek verirsek ID si 2 olan kullanıcının yüzüncü görüntüsü aşağıdaki şekilde kaydedilecektir.
	
“User.2.100.jpg”

Oluşturulan veri tabanı daha sonraki işlemde kullanılacaktır.
### Model Eğitme
İmutils, face_recognition, Pickle, Opencv, OS kütüphaneleri kullanılmıştır. Kodda ilk olarak daha önce oluşturulmuş olan veri setimiz ekleniyor. Veri setindeki ID numaralarına göre resimler ayrılıyor. Resimler algoritmada kullanılabilmek için BGR’dan RGB ye çevriliyor. Face_recognition kütüphanesinin face_locations fonksiyonu kullanılarak her bir resimdeki yüzler ilk olarak tespit ediliyor.

### Eğitilen Model ile Yüz Tanıma Sistemi
Daha önce eğitilmiş olan model Pickle kütüphanesinin load metodu yardımıyla yükleniyor. OpenCV ile kameradan elde edilen frameler BGR formattan RGB formata çevriliyor. face_recognition.face_encodings bu şekilde frameler encode ediliyor. Bu elde edilen veriler ile daha önceden eğitilmiş modeldeki encode edilmiş veriler, face_recognition.compare_faces yardımıyla karşılaştırılır. Karşılaştırma sonucu eşleşme bulunduğu zaman, bulunan kişinin ID si alınarak ID ye karşılık gelen isim aşağıdaki şekilde ekrana bastırılır. Tespit edilen yüzde left, top, right, bottom fonksiyonları kullanılarak bulunan yüzün x ve y eksenlerindeki değerleri bulunur. Yüzün dikdörtgen içine alınmasıda bu değerlerin cv2.rectangle fonksiyonuyla çizdirilmesiyle sağlanır.

## 3_Duygu_tanıma_beş_katman
Beş katmanlı yapay sinir ağı modeli kullanılmıştır. Google Colab üzerinde ücretsiz GPU olarak Tesla K80 seçilerek eğitim gerçekleştirilmiştir. Eğitimde Keras ve Tensorflow ağırlıklı kütüphaneler kullanılmıştır. Veri seti olarak Fer 2013 kullanılmıştır. Kodumuzda ilk olarak Fer2013 veri setinden alınan veriler train ve test olarak ikiye ayrılıyor. Ayrılan train ve test verilere ön işleme uygulanıyor. Ön işlemeden sonra “x_test”, “y_test”, “x_train”, “y_train” olarak daha sonra kullanabileceğimiz veri grubunu elde ediyoruz. Bu verilerin shapeleri (48,48,1) olarak ayarlanmıştır.

Model oluşturma aşamasında altı adet konvolüsyon katmanı kullanıldı. Yedinci katma tam bağlantı katmanı ve son katman ise çıkış katmanı olarak belirlendi.

## 3_Duygu_tanıma_hazır
Kodda kullanılan CNN yapısında konvolüsyon katmanında VGG-face kütüphanesinden alınan konvolüsyon katmanları eklenmiştir. Model olarak VGG16 seçilmiştir.

VGG16, K. Simonyan ve A. Zisserman tarafından "Büyük Ölçekli Görüntü Tanıma için Çok Derin Evrişimli Ağlar " makalesinde tavsiye edilen evrişimli bir sinir ağı modelidir. 

VGG16’ nın mimari özeti aşağıdaki şekil 3-15’deki gibidir.

![alt text](https://github.com/Berkayyldrm/Guvenli-Surus-Sistemi/blob/main/img/Resim4.png)
Şekil 3 15 VG16 Mimari Özellikleri

İnput_shape 197, 197, 3 olarak ayarlanmıştır. Vgg face kütüphanesinden ki konvolüsyon katmanlarından sonra modele Fully connect katmanları eklenmiştir. Son katmanda da yedi tane duygu tahmin edeceğimiz için yedi olarak ayarlanması gerekiyor.

Tek bir modelin eğitimi GPU kullanılsa bile haftalar sürdüğü için bu modelin eğitiminin çıktısı hazır olarak kullanılmıştır. Modelin kamera yardımıyla test edilmesi, test için hazırlanan kodda önceden eğitilmiş model ve yüz tespiti için Haar Cascade kullanılmıştır.

OpenCV yardımıyla kameradan elde edilen frameler üzerinde işlemler yapılmaktadır. İlk olarak kameradaki frameler üzerinden yüzler tespit edilmektedir. Frameler (197,197,3) shapeine çevrilmektedir. Gerekli ayarlamalardan sonra elde edilen frameler ile modeldeki verilere göre tahmin gerçekleşmektedir. Tahmin olarak hangi duyguya ne kadar yakın olduğu oransal olarak verilmektedir. En yüksek oranın olduğu duygu ise kullanıcıya çıktı olarak verilmektedir.

# Final_Project
	Aşağıdaki link kullanılarak eğitimlerde kullanılan veri setlerine, modellere, ve dlib için gerekli olan shape predictor modeline ulaşabilirsiniz.
	https://drive.google.com/drive/folders/1OkmOYf46hUi7iC9lusrLaQm5qCdmzhji?usp=sharing

	Dosyada bulunan VGG16 modeli Final_Project 'nun içine atılmalı
	Dosyada bulunan shape_predictor, Final_Project 'nun içine atılmalı

Klasörde bulunan 1, 2, 3, 4, Admin_log, Sifre dosyaları Pyqt5 ile oluşturulmuş masaüstü uygulama dosyalarıdır.

AdminLog.csv ve Logins.csv giriş id ve şifrelerinin kaydedildiği dosyadır.

face_rec, model.kayit, VGG16, xx dosyaları eğitilmiş modellerin dosyalarıdır.

ANA_KOD dosyası 1_Yorgunluk_dlib ile implement edilmiştir.

ANA_KOD2 dosyası 1_Yorgunluk_Hog_svm ile implement edilmiştir.
