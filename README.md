# Guvenli-Surus-Sistemi
Mezuniyet projesi olarak gerçekleştirilmiştir. Projede, araç sürücüsünün yorgunluk, güvenlik ve duygu durumu için makine öğrenmesi algoritmaları kullanılmıştır. Pyqt5 kullanılarak algoritmalar tek programda birleştirilmiştir.

# Preparation of Project
## 1_Yorgunluk_Dlib
Sistemde Opencv, Dlib, Scipy kütüphaneleri kullanılmıştır. Kodda ilk olarak OpenCV kütüphanesiyle kameradan görüntü alınmaya başlıyor. Dlib kütüphanesinden get_frontal_face_detector ve shape_predictor classlarından obje tanımlanıyor. Video görüntüleri, tek tek resim framelerinden oluştuğundan sonsuz döngü oluşturuluyor. Her bir döngüde fps oranına göre frameler tek tek alınıp kodun ileriki aşamalarında işleniyor. Alınan frameler BGR renk formatındadır. Görüntüler üzerinde işlem yapabilmek için BGR renk formatı GRAY renk formatına çevrilmiştir. Gray scale deki görüntü kullanılarak get_frontal_face_detector ile görüntüdeki yüzler tespit edilmektedir. Tespit edilen yüzler kullanılarak shape_predictor kullanılarak yüzün belirli noktaları çıkarılmaktadır. (Şekil 3-8)
 
Şekil 3 8 [My image](Berkayyldrm.github.com/repository/img/Resim1.png) Face Landmark
	Yukarıda belirtilen noktalardan 36-42 arası sol gözü, 42-48 arası sağ gözü, 48-60 arası ağzı temsil etmektedir.
	Bu noktalar kullanılarak sol göz, sağ göz ve ağız için her biri için ayrı döngüde bu noktaların konumları ekranda çizilerek kullanıcıya gösterilecektir. Bu noktaların x, y koordinatları daha önceden tanımlanan dizilerin içine kaydedilecektir. Gözler ve ağız için mesafe ölçme fonksiyonları tanımlanmıştır. Örnek olarak Şekil 3-9’daki sol göz için hesaplamalar aşağıdaki gibidir.
 
Şekil 3 9 Sol Göz Landmarkı
Gözlerin açıklık oranlarının bulunması aşağıdaki formüle göre yapılmıştır.
    Göz açıklık oranı=(A+B)/(2*C)(denklem 3.1)
A:37-41 arası mesafe
B:38-40 arası mesafe
C:36-39 arası mesafe
Ağzın açıklık oranının bulunması aşağıdaki formüle göre yapılmıştır.
 
    Şekil 3 10 Ağız Landmarkı
    Ağız açıklık oranı=(A+B)/(2*C)(denklem 3.2)
A:50-58 arası mesafe
B:52-56 arası mesafe
C:48-54 arası mesafe
Mesafeler ölçülürken Scipy kütüphanesinin spatial fonksiyonun distance metodu kullanılmıştır. Distance metodunun Euclidean özelliği ile mesafeler ölçülmüştür.
Sağ ve sol gözün açıklık değerleri ayrı bir fonksiyonda ortalamaları alınarak kullanılmıştır. Bu ortalama değere göre gözün açıklık kapalılık durumu bir tane kesim değerine göre belirlenmiştir. Kesim değeri 0.24 olarak alınmıştır. 0.24 üzeri değerler gözün açık olduğunu, 0.24 altı değerler ise gözün kapalı olduğunu belirtmektedir. Gözün açık veya kapalı olduğu durumlar bir tane diziye kaydedilmektedir. Bu diziden de son 150 frame deki değerler alınmaktadır. Sistemin fps değeri 10 olduğundan yani son 15 saniyedeki değerler alınmaktadır.
Bu alınan değerler Perclose fonksiyonuna gönderilmektedir. Bu fonksiyonda Perclose algoritması bulunmaktadır. Perclose algoritması[15] gözdeki yorgunluk tespitini göz durumlarına kullanarak tespit eden bir algoritmadır. Aşağıda algoritma verilmiştir.
Toplam frame = açık gözün olduğu frame + kapalı gözün olduğu frame
Perclose değeri = kapalı gözün olduğu frame * 100 / Toplam frame
Perclose değerinin eşik değeride 60 olarak belirlenmiştir. 60 üzeri değerde sürücü “Yorgun” olarak belirtiliyor. 60 altında değerde ise sürücü “Normal” olarak belirtiliyor. Bu elde edilen sonuç masaüstü uygulamasında gösteriliyor.
Esneme tespit fonksiyonunda ise önceden elde edilmiş mesafe değerini kullanılmıştır. Bu mesafe değerinin eşik değeri 60 olarak belirtilmiştir. 60 üzeri değerler esneme gerçekleşiyor olduğunu belirtiliyor. 60 altı değerler ise esneme olmadığını belirtmektedir. Değerin 60 üzeri olduğu durumlarda counter devreye girmektedir. Her 60 üzeri framede counter bir artmaktadır. Sistemin fps i 10 olduğundan 1 saniyelik esneme, counter’ı on arttırmaktadır. Esneme tespit fonksiyonuna 1 adet te timer eklenmiştir. Her otuz saniyede bir timer sıfırlanmaktadır ve sürücünün esneme durumuna göre yorgun veya normal olduğu masaüstü uygulamasında gösterilmektedir. Esneme durumuna göre yorgunluk tespitide counter değerinin 30 un altında olup olmadığına göre yapılmaktadır. Yani sürücü otuz saniyede üç kere esnediyse sistem sürücüye yorgunluktan dolayı uyarı vermektedir.
