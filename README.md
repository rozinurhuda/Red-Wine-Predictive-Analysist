# Laporan Proyek Machine Learning - Achmad Rozi Nurhuda

## Domain Proyek

*Red wine* atau biasa disebut anggur merah memiliki tempat istimewa dalam dunia kuliner dan minuman karena keunikan rasa dan karakteristiknya. Hal itu tergatung pada berbagai faktor seperti varietas anggur, iklim tempat tumbuhnya, dan teknik produksi. Keunikan rasa dan karakteristik ini mencerminkan kombinasi kompleks dari faktor kimia, fisik, dan organoleptik yang terlibat dalam proses pembuatan anggur merah.

Salah satu keunikan anggur merah terletak pada tingkat keasaman dan keseimbangannya. Tingkat asam pada anggur merah memberikan kecerahan dan kesegaran pada rasa, serta berperan dalam menjaga stabilitas anggur. Kandungan asam tartarat dan malat memberikan anggur merah karakteristik rasa yang berbeda -beda, mulai dari asam yang menyegarkan hingga asam yang lembut.

Selain itu, anggur merah juga memiliki kandungan alkohol yang memberikan pengaruh signifikan terhadap rasa dan kekayaan anggur. Kandungan alkohol yang tepat dapat memberikan kehangatan dan kelengkapan rasa, serta memberikan tekstur yang lembut pada anggur merah.

Selanjutnya, anggur merah juga dipengaruhi oleh tannin, senyawa polifenol alami yang terkandung dalam kulit anggur. Tannin memberikan struktur dan astringensi pada anggur merah, serta memberikan rasa pahit dan kekeringan yang menyatu dengan rasa buah dan kompleksitas anggur.

Selain faktor-faktor tersebut, karakteristik anggur merah juga dipengaruhi oleh tingkat pH, konsentrasi aroma, dan mineral yang terkandung di dalamnya. Semua faktor ini berkontribusi dalam menciptakan berbagai profil rasa anggur merah yang unik dan kompleks. [1]

Membuat keputusan yang tepat dalam memilih jenis *red wine* yang cocok dengan preferensi dan selera seseorang dapat menjadi tugas yang menantang. Oleh karena itu, proyek ini akan menggunakan teknik predictive analysis dengan menggunakan data historis red wine yang telah terkumpul untuk membangun model yang dapat memprediksi dan merekomendasikan jenis red wine yang sesuai dengan preferensi individu.

Melalui pendekatan *machine learning*, kita akan menggunakan algoritma dan metode yang tepat untuk menganalisis pola dalam data *red wine*, termasuk faktor seperti varietas anggur, wilayah produksi, musim panen, nilai sensorik, dan faktor lain yang mempengaruhi kualitas dan karakteristik *red wine*. Dengan memanfaatkan teknik seperti regresi, klasifikasi, atau pengelompokan, model *machine learning* akan belajar dari data yang ada dan mengidentifikasi hubungan dan tren yang signifikan. Hasilnya akan memberikan pemahaman yang lebih baik tentang preferensi individu dan memberikan rekomendasi *red wine* yang cocok berdasarkan preferensi pengguna.

Dengan menggabungkan keahlian dalam bidang anggur *red wine* dan teknik *machine learning*, proyek ini memiliki potensi untuk menghasilkan wawasan baru dan solusi yang inovatif dalam dunia anggur, yang akan memberikan nilai tambah yang signifikan bagi pecinta *red wine* dan para pemangku kepentingan dalam industri ini.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara *preprocessing* pada data *Quality Red Wine* yang akan digunakan untuk membuat model yang baik?
- Bagaimana cara memilih/membuat model yang terbaik untuk memprediksi kualitas <em> red wine </em>?
- Berapa hasil akurasi prediksi dari model algoritma?
- Faktor apa saja yang mempengaruhi kualitas anggur merah?
- Jenis *red wine* mana yang paling dapat dijual menurut preferensi pelanggan?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Melakukan *preprocessing* data sehingga data tersebut siap untuk di latih oleh model *Machine Learning*
- Menggunakan *LazyPredict* untuk mencari 5 Algoritma <em> Machine Learning </em> terbaik.
- Mengevaluasi akurasi dari setiap model algoritma tersebut.  
- Memahami faktor-faktor yang memengaruhi kualitas anggur merah: Pemahaman mendalam tentang faktor-faktor kimia, fisik, dan organoleptik yang memengaruhi kualitas anggur merah akan membantu mengidentifikasi atribut-atribut yang perlu dianalisis dan diprediksi.

- Memahami preferensi pelanggan: Mengetahui preferensi pelanggan terkait dengan karakteristik dan keunikan rasa anggur merah dapat membantu dalam mengembangkan model prediktif yang dapat memprediksi preferensi pelanggan dan menghasilkan rekomendasi yang disesuaikan.

Setelah *goals* dicapai diharapkan dapat memberikan manfaat yang berarti bagi pecinta anggur *red wine* dan industri anggur secara keseluruhan. Dengan menggunakan analisis prediktif, para konsumen akan dapat memilih *red wine* yang sesuai dengan preferensi mereka dengan lebih percaya diri. Di sisi lain, produsen dan pengecer anggur dapat menggunakan wawasan dari model *machine learning* ini untuk meningkatkan strategi pemasaran, mengoptimalkan produksi, dan menghadirkan pengalaman yang lebih memuaskan bagi pelanggan. Dalam rangka mencapai tujuan ini, proyek ini akan melibatkan eksplorasi data, pemrosesan data, pemodelan *machine learning*, dan evaluasi kinerja model untuk mencapai hasil yang optimal.

## Data Understanding
Data yang digunakan adalah data yang berasal dari kaggle, kumpulan data ini terkait dengan varian anggur merah "*Vinho Verde*" Portugis.[2] Karena masalah privasi dan logistik, hanya variabel fisikokimia (*input*) dan sensorik (*output*) yang tersedia (misalnya tidak ada data tentang jenis anggur, merek anggur, harga jual anggur, dll).

Berikut ini informasi mengenai fitur dataset:
1. ***fixed acidity*** - Jumlah asam tartarat dalam g/L, meskipun fixed acidity umumnya dalam anggur termasuk asam malat, sitrat, dan suksinat juga.
2. ***volatile acidity*** - Jumlah asam asetat (cuka) dalam g/L. Volatile acidity yang lebih umum juga termasuk laktat, formik, butirat, dan propionat. Asam ini terkait dengan pembusuan anggur.
3. **citric acid** - Jumlah asam sitrat dalam g/L. Asam sitrat biasanya hadir dalam anggur tetapi dapat ditambahkan ke anggur untuk meningkatkan keasaman.
4. ***residual sugar*** - Biasanya jumlah gula alami dalam g/L yang tersisa dalam anggur setelah proses fermentasi selesai. Beberapa negara mengizinkan tambahan gula untuk ditambahkan, tetapi praktik ini tidak disukai oleh para kritikus.
5. ***chlorides*** - Jumlah sodium klorida dalam g/L.
6. ***free sulfur dioxide*** - Jumlah sulfit yang tersedia untuk bereaksi dalam mg/L. Sulfit (sulfur dioksida atau SO2) sering ditambahkan ke anggur sebagai pengawet, tetapi beberapa juga terjadi secara alami.
7. ***total sulfur dioxide*** - Jumlah total sulfit bebas dan sudah bereaksi (terikat) dalam mg/L.
8. ***density*** - Terukur dalam g/ml.
9. ***pH*** - Pengukuran keasaman anggur (pH lebih rendah lebih asam)
10. ***sulphates*** - Bentuk lain dari belerang alami (SO4) yang bergantung pada komposisi tanah tempat anggur ditanam.
11. ***alcohol*** - Persentase alkohol berdasarkan volume.
12. ***quality*** - nilai antara 0 sampai dengan 10.

*Overview Data* :

- *Datasets Name :  Red Wine Quality*
- *Overall Columns:*
    - *Valid : 1599*
    - *MissMatched : 0*
    - *Missing : 0*
- *Source : UCI Machine Learning*
- *Link : https://archive.ics.uci.edu/ml/datasets/wine+quality*
- *License : Database: Open Database, Contents: Database Contents*
- *Inspiration : Use machine learning to determine which physiochemical properties make a wine 'good'!*

Dari beberapa fitur yang terdapat pada dataset *Red Wine Quality*, tentunya ada hal yang sangat mempengaruhi kualitas dari anggur merah. Fitur-Fitur yang mempengaruhi itu dapat dilihat dari tabel dibawah ini:

1. Fitur yang pertama adalah *Quality vs pH* terlihat pada gambar dibawah ini.

![quality vs ph](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/36979e24-1d52-45ab-95f5-44e6f990856b)

Gambar 1. Diagram batang *Quality vs pH*

Gambar diatas menunjukkan bahwa semakin kecil *pH* kualitas anggur merah akan semakin bagus, namun tidak terlalu signifikan


2. Fitur kedua adalah *Quality vs Fixed Acidity* terlihat pada gambar dibawah ini.

![quality vs fixed acidity](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/a53b9f26-43cf-42db-bf22-3c974bcf4e33)

Gambar 2. Diagram batang *Quality vs Fixed Acidity*

Gambar diatas tidak relevan untuk digunakan sebagai acuan karena pada nilai *quality* 4 menunjukkan angka paling kecil, namun pada *quality* 3 angka diatas dari 5 dan 6. Begitu pula pada nilai tertinggi 8, angka menunjukkan lebih kecil dari pada nilai *quality* 7.

3. Fitur ketiga adalah *Quality vs Volatile Acidity* terlihat pada gambar dibawah ini.

![quality vs volatile acidity](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/b122a4ae-51f1-4cc0-9ac7-0d78b819dfdf)

Gambar 3. Diagram batang *Quality vs Volatile Acidity*

Gambar diatas menunjukkan bahwa semakin kecil kadar *volatile acidity* pada anggur merah, maka kualitasnya akan semakin baik.

4. Fitur keempat adalah *Quality vs Citric Acid* terlihat pada gambar dibawah ini.

![quality vs citrid acid](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/8e95f652-1c41-4de4-a24c-434248294af6)

Gambar 4. Diagram batang *Quality vs Citric Acid*

Terlihat jelas bahwa kandungan *citric acid* atau sitrat sangat berpengaruh pada kualitas anggur merah. Semaking tinggi kandungan asam sitrat kualitasnya semakin baik.

5. Fitur kelima adalah *Quality vs Residual Sugar* terlihat pada gambar dibawah ini.

![quality vs residual sugar](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/5c39defd-e879-49b7-88a6-82aac6d0a3a9)

Gambar 5. Diagram batang *Quality vs Residual Sugar*.

Seperti halnya *fixed acidity*, *residual sugar* pun juga tidak relevan dalam mempengaruhi kualitas anggur merah. Terlihat dari grafiknya tidak konsisten dan naik turun disetiap nilai kualitasnya.

6. Fitur keenam adalah *Quality vs Chlorides* terlihat pada gambar dibawah ini.

![quality vs chlorides](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/46d3f910-6b25-4c02-81fc-5a30d6ccb534)

Gambar 6. Diagram batang *Quality vs Chlorides*.

Dari gambar diatas dapat disimpulkan bahwa semakin banyak kandungan *chlorides* atau sodium klorida dalam anggur merah makan kualitasnya akan semakin buruk.

7. Fitur ketujuh adalah *Quality vs Free Sulfur Dioxide* terlihat pada gambar dibawah ini.

![quality vs free sulfur dioxide](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/8c3b087e-4088-4310-9738-f5d5a23afa5c)

Gambar 7. Diagram batang *Quality vs Free Sulfur Dioxide*.

Gambar diatas menunjukkan bahwa pengaruh kandungan *free sulfur dioxide* mencolok pada kualitas 5 dan 6. Hal ini bisa diasumsikan bahwa pecinta anggur dikelas menengah cenderung menyukai anggur dengan kandungan *free Sulfur dioxide* yang tinggi, mengingat dari jumlah data yang diambil, terbanyak berada pada nilai kualitas 5 dan 6.

8. Fitur kedelapan adalah *Quality vs Total Sulfur Dioxide* terlihat pada gambar dibawah ini.

![quality vs total sulfur dioxide](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/0eb78bc8-8d93-4e2d-bd5d-8976e70f56c7)

Gambar 8. Diagram batang *Quality vs Total Sulfur Dioxide*.

Dari gambar diatas menunjukkan semakin jelas jika kandungan *total sulfur dioxide* terbanyak pada nilai kualitas 5. Sama halnya pada asumsi sebelumnya bahwa anggur merah dikelas menengah memiliki banyak kandungan sulfit, baik itu dari yang tersedia untuk bereaksi ataupun total kandungan sulfinya.

9. Fitur kedelapan adalah *Quality vs Density* terlihat pada gambar dibawah ini.

![quality vs density](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/59463e12-daf7-4bdf-ae37-6ca97eb04f5d)

Gambar 9. Diagram batang *Quality vs Density*.

Gambar diatas menunjukkan bahwa tidak ada pengaruhnya kualitas anggur merah dengan kandungan *density* didalamnya.

10. Fitur ke sepuluh adalah *Quality vs Sulphates* terlihat pada gambar dibawah ini.

![quality vs sulphates](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/dfa1a992-bf6c-47c2-b32e-b78e65d95b9e)

Gambar 10. Diagram batang *Quality vs Sulphates*.

Kandungan sulfat pada anggur merah sangat mempengaruhi kualitas anggur merah, didukung dengan gambar diatas. Semakin banyak kandungan sulfat pada anggur merah semakin baik pula kualitasnya.

11. Fitur kesebelas adalah *Quality vs Alcohol* terlihat pada gambar dibawah ini.

![quality vs alcohol](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/77042ab0-7114-4cb7-963b-82634a0a9994)

Gambar 11. Diagram batang *Quality vs Alcohol*

Gambar diatas menunjukkan bahwa kandungan alcohol dalam anggur merah sangat mempengaruhi kualitasnya.

Dari kesebelas fitur diatas terdapat 5 fitur yang sangat berpengaruh terhadap kualitas anggur merah, fitur-fitur tersebut adalah sebagai berikut:

- **Chlorides**
- **Citric Acid**
- **Volatile Acidity**
- **Sulphates**
- **Alcohol**

### Analisa Deskriptif

Tabel 1. *Generative Describe Statistics*

|                      |   count |       mean |         std |     min |     25% |      50% |       75% |       max |
|----------------------|---------|------------|-------------|---------|---------|----------|-----------|-----------|
| fixed acidity        |    1599 |  8.31964   |  1.7411     | 4.6     |  7.1    |  7.9     |  9.2      |  15.9     |
| volatile acidity     |    1599 |  0.527821  |  0.17906    | 0.12    |  0.39   |  0.52    |  0.64     |   1.58    |
| citric acid          |    1599 |  0.270976  |  0.194801   | 0       |  0.09   |  0.26    |  0.42     |   1       |
| residual sugar       |    1599 |  2.53881   |  1.40993    | 0.9     |  1.9    |  2.2     |  2.6      |  15.5     |
| chlorides            |    1599 |  0.0874665 |  0.0470653  | 0.012   |  0.07   |  0.079   |  0.09     |   0.611   |
| free sulfur dioxide  |    1599 | 15.8749    | 10.4602     | 1       |  7      | 14       | 21        |  72       |
| total sulfur dioxide |    1599 | 46.4678    | 32.8953     | 6       | 22      | 38       | 62        | 289       |
| density              |    1599 |  0.996747  |  0.00188733 | 0.99007 |  0.9956 |  0.99675 |  0.997835 |   1.00369 |
| pH                   |    1599 |  3.31111   |  0.154386   | 2.74    |  3.21   |  3.31    |  3.4      |   4.01    |
| sulphates            |    1599 |  0.658149  |  0.169507   | 0.33    |  0.55   |  0.62    |  0.73     |   2       |
| alcohol              |    1599 | 10.423     |  1.06567    | 8.4     |  9.5    | 10.2     | 11.1      |  14.9     |
| quality              |    1599 |  0.13571   |  0.342587   | 0       |  0      |  0       |  0        |   1       |

Berikut adalah hasil analisis dari tabel diatas.
  - Sampel sekitar 1599 data
  - Nilai kualitas terendah adalah 3 dan nilai kualitas tertingginya adalah 8.
  - Nilai rata-rata kualitas data diatas adalah 5,64.

### Visualization

Berikut ini adalah gambar diagram batang banyaknya data *quality* dengan range nilai 3 sampai dengan 8. Setelah ini data akan dibagi menjadi 2 bagian, nilai 3 sampai dengan 6.4 akan diasosiasikan sebagai anggur kurang baik dan nilai 6.5 sampai dengan 8 akan diasosiasikan sebagai anggur baik.

![quality-count](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/4c9fae0f-6d8a-4a51-8995-0a601a202127)

Gambar 12. Diagram batang jumlah data pada *quality*

Gambar diagram batang diatas menunjukkan bahwa pada data *"Red Wine Quality"* data terbanyak adalah pada nilai kualitas 5 dan 6, masing-masing memiliki jumlah 681 dan 638 data.

## Data Preparation

Teknik *Data preparation* yang dilakukan adalah sebagai berikut:

- Mengelompokan data *quality* kedalam 2 bagian, dengan cara mengganti data *quality* dengan 0 dan 1, sesuai ketentuan diatas untuk data **0** berarti anggur kurang baik (*Bad Wine*) dan **1** berarti anggur baik (*Good Wine*)
- Menyeimbangkan data antara jumlah data *Bad Wine* dengan *Good Wine* dengan menggunakan *imblearn, oversampling dan undersampling*.
- Membagi dataset menjadi data latih (*train*) dan data uji (*test*) dengan menggunakan *TrainTestSplit*.

Berikut adalah hasil akhir dari *data preparation* yang dapat dilihat dari gambar dibawah ini.

![0 and 1-count](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/3b783cec-cd66-4a6c-b2bc-28e47141d3de)

Gambar 13. Diagram batang jumlah data quality sebalum dilakukan Penyeimbangan

Diagram diatas menunjukkan bahwa perbandingan antara jumlah *bad wine* and *good wine* tidak seimbang, terlalu banyak data pada *bad wine* dari pada *good wine*. Hal ini tidak akan membuat proses prediksi tidak optimal, maka dari itu perlu dilakukan data *balancing* pada data tersebut melalui proses *undersampling* dan *oversampling*

- *Undersampling*

*Undersampling* adalah teknik di mana kita mengurangi jumlah sampel dari kelas mayoritas sehingga seimbang dengan jumlah sampel kelas minoritas. Dengan mengurangi jumlah sampel dari kelas mayoritas, kita dapat mengurangi bias yang mungkin terjadi dalam model yang dilatih pada data yang tidak seimbang. Dalam modul *imblearn*, terdapat beberapa metode *undersampling* yang dapat digunakan, seperti *"RandomUnderSampler"*, *"NearMiss"*, dan *"TomekLinks"* [3].

- *Oversampling*

*Oversampling* adalah teknik di mana kita meningkatkan jumlah sampel dari kelas minoritas sehingga seimbang dengan jumlah sampel kelas mayoritas. Dengan meningkatkan jumlah sampel kelas minoritas, kita dapat memperluas variasi data yang tersedia untuk melatih model dan mengurangi risiko overfitting pada data yang tidak seimbang. Dalam modul imblearn, terdapat beberapa metode *oversampling* yang dapat digunakan, seperti *"RandomOverSampler"*, *"SMOTE"* (*Synthetic Minority Oversampling Technique*), dan *"ADASYN"* (*Adaptive Synthetic Sampling*) [3]. 

Kombinasi dari kedua pendekatan ini juga dapat digunakan untuk mencapai keseimbangan yang optimal dalam data yang tidak seimbang. Berikut ini adalah gambar diagram lingkaran yang menunjukkan hasil proses *undersampling*.


![undersampling](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/b451dd9a-1806-43c8-9eb3-036d107609f1)


Gambar 14. Diagram lingkaran hasil *undersampling*.

Gambar diatas menunjukkan bahwa jumlah sampel dari kelas mayoritas dalam hal ini *bad wine* telah dikurangi jumlahnya sehingga seimbang dengan data sampel *good wine*.

Selanjutnya, berikut ini adalah gambar diagram lingkarang yang menunjukkan hasil proses *oversampling*.

![oversampling](https://github.com/rozinurhuda/Red-Wine-Predictive-Analysist/assets/11625567/da676114-290a-4286-9884-336c1798d010)

Gambar 15. Diagram lingkaran hasil *oversampling*

Gambar diatas menunjukkan bahwa hasil dari proses *oversampling* telah mendapatkan hasil yang seimbang.

## Modelling

Setelah berhasil mendapatkan dataset yang seimbang, barulah kita melakukan pemodelan untuk dapat mendapatkan model algoritma terbaik. berikut adalah langkah-langkah dalam melakukan pemodelan.

- Melakukan *Handling null value*, bisa dengan *mean, median*.
- Melakukan pembagian dataset menjadi dua bagian dengan rasio 8:2 / 80% untuk *train* dan 20% untuk *test*.
- Menggunakan pustaka python yaitu *Lazy Predict* untuk mendapatkan 5 algoritma  <em> Machine Learning </em> terbaik.
- Melakukan prediksi terhadapa model dan menghitung menggunakan beberapa metrik seperti *precission, recall, f1-score*.
    - Berikut adalah rumus untuk menghitung <em> precission </em>:
        
        $$ precission = {TP \over TP + FP} $$
        
    - Berikut adalah rumus untuk menghitung <em> recall </em>:
    
        $$ recall = {TP \over TP + NP} $$

      - ***True Negative* (TN)** : Model meprediksi data ada di kelas negatif dan yang sebenarnya data memang ada di kelas Negatif.
      - ***True Positive* (TP)** : Model memprediksi data ada di kelas positif dan sebenarnya data memang ada di kelas positif
      - ***False Negative* (FN)** : Model memprediksi data ada di kelas Negatif, namun yang sebenarnya data di kelas Positif.
      - ***False Positive* (FP)** : Model memprediksi data ada di kelas positif namun yang sebenarnya data ada di kelas Negatif.
    - Berikut adalah rumus untuk menghitung *F1-Score*:

        $$ {1 \over F1} = {1 \over 2} \left( { 1 \over precission} \right) + \left({1 \over recall}\right) $$

    - Rumus-rumus diatas dapat dihitung langsung menggunakan library python yaitu sklearn metrics.


Dari sampel data yang sudah terlihat jelas bahwa permasalahan yang harus diselesaikan adalah masalah klasifikasi. Untuk menyelesaikan masalah klasifikasi tentunya banyak sekali algoritma yang bisa digunakan, namun hal ini bisa diketahui dengan memanfaatkan salah satu pustaka *LazyClassifier* dari *Lazypredict*. *LazyClassifier* adalah pustaka yang memungkinkan kita untuk melatih berbagai model klasifikasi secara otomatis tanpa perlu mengatur parameter secara manual. Ini adalah alat yang berguna untuk mendapatkan pemahaman awal tentang kinerja beberapa model pada dataset.

Dari hasil yang didapat dari *LazyClassifier* akan dipilih 5 algoritma terbaik dan kemudian algoritma-algoritma tersebut akan dievaluasi kembali sesuai dengan model algoritma masing-masing. Semua data yang digunakan dalam proses pemodelan, menggunakan rasio data 8:2 yaitu 80% data digunakan untuk pelatihan dan 20% data digunakan untuk pengujian menggunakan *'train_test_split'*.

### Lazy Predict

Langkah pertama yaitu mencari 5 algoritma terbaik menggunakan salah satu pustaka *Lazy Predict* yaitu *LazyClassiffier*.
Hasil dari *Lazy Predict* dapat dilihat dari tabel dibawah ini.

Tabel 2. Hasil Perbandingan Model Menggunakan *Lazy Predict*.

| Model                  |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
|------------------------|------------|---------------------|-----------|------------|--------------|
| ExtraTreesClassifier   |   0.980108 |            0.9811   |  0.9811   |   0.980121 |    0.310411  |
| RandomForestClassifier |   0.962025 |            0.963918 |  0.963918 |   0.962046 |    0.534989  |
| XGBClassifier          |   0.9566   |            0.958763 |  0.958763 |   0.956617 |    0.291812  |
| ExtraTreeClassifier    |   0.9566   |            0.958763 |  0.958763 |   0.956617 |    0.0208921 |
| BaggingClassifier      |   0.952984 |            0.955326 |  0.955326 |   0.952996 |    0.136845  |

Dapat dilihat bahwa model terbaik yang didapat dari *Lazy Predict* adalah *ExtraTressClassifier*

Tahapan selanjutnya adalah mengevaluasi semua hasil diatas dan kemudian hasil akurasi akan ditampilkan pada *final report* diakhir sub bab ini.

### Extra Trees Classifier



*ExtraTreesClassifier* adalah sebuah algoritma pembelajaran mesin yang digunakan untuk klasifikasi dalam konteks pembelajaran berbasis pohon. Ini merupakan variasi dari algoritma *Random Forest*. *ExtraTreesClassifier* menggunakan konsep *ensemble learning*, di mana beberapa model pohon keputusan (*decision trees*) digabungkan untuk membuat prediksi akhir.

Keunikan dari *ExtraTreesClassifier* terletak pada penggunaan metode pengambilan keputusan yang berbeda dalam pembentukan setiap pohon keputusan dalam *ensemble*. Pada setiap split dalam pembangunan pohon, *ExtraTreesClassifier* secara acak memilih subset acak dari fitur-fitur yang tersedia dan menggunakan nilai ambang batas acak untuk membuat pemisahan. Ini berbeda dari algoritma *Random Forest*, yang menggunakan metode pemilihan terbaik untuk pemisahan.

Pada model *ExtraTreesClassifier* ini parameter-parameter yang digunakan adalah:
* *n_estimators=100* = 100
* *random_state* = 42

Setelah mengatur parameternya, model dilatih dengan memanggil metode *'fit'* menggunakan data latih. Setelah model dilatih, dilakukan prediksi pada data uji dengan memanggil metode *'predict'*. Akurasi hasil prediksi dihitung menggunakan '*accuracy_score*' dengan membandingkan label sebenarnya (y_test) dengan prediksi (y_pred). setelah dilakukan prediksi pada data uji, langkah selanjutnya adalah mengevaluasinya menggunakan *'cross_val_score'* dengan parameter cv=10.

Tabel 3. Hasil prediksi untuk model *Extra Trees Classifier*.

|               |   precision   |   recall   |   f1-score   |   support   |
|---------------|---------------|------------|--------------|-------------|
|       0       |      1.00     |    0.96    |     0.98     |     291     |
|       1       |      0.96     |    1.00    |     0.98     |     262     |
|               |               |            |              |             |
| accuracy      |               |            |     0.98     |     553     |
| macro avg     |      0.98     |    0.98    |     0.98     |     553     |
| weighted avg  |      0.98     |    0.98    |     0.98     |     553     |

### DecisionTreeClassifier

*DecisionTreeClassifier* adalah sebuah algoritma yang digunakan dalam *machine learning* untuk melakukan klasifikasi pada data. Algoritma ini membangun model prediksi dalam bentuk pohon keputusan, di mana setiap simpul (*node*) pada pohon merepresentasikan suatu pengujian pada fitur-fitur data, setiap cabang (*branch*) merepresentasikan hasil dari pengujian tersebut, dan setiap daun (*leaf*) merepresentasikan label kelas.

Proses pembentukan pohon keputusan dimulai dengan memilih fitur terbaik yang membagi data menjadi dua *subset* berdasarkan pengukuran keberagaman (misalnya, indeks Gini atau entropi). Kemudian, algoritma secara rekursif membagi *subset-subset* tersebut hingga mencapai kondisi berhenti yang ditentukan sebelumnya, seperti kedalaman maksimum pohon atau jumlah sampel minimum di setiap daun.


Pada model *DecisionTreeClassifier* parameter-parameter yang digunakan adalah:
* *max_depth*: 7
* *max_feature*: *none*
* *min_samples_leaf': 2
* *min_sample_split': 2

Setelah mengatur parameternya, model dilatih dengan memanggil metode 'fit' menggunakan data latih. Setelah model dilatih, dilakukan prediksi pada data uji dengan memanggil metode '*predict*'. Akurasi hasil prediksi dihitung menggunakan '*accuracy_score*' dengan membandingkan label sebenarnya (y_test) dengan prediksi (y_pred). setelah dilakukan prediksi pada data uji, langkah selanjutnya adalah mengevaluasinya menggunakan *'cross_val_score'* dengan parameter cv=10.


Tabel 4. Hasil prefiksi untuk model *Decision Tree Classifier*.

|               |   precision   |   recall   |   f1-score   |   support   |
|---------------|---------------|------------|--------------|-------------|
|       0       |      0.94     |    0.86    |     0.90     |     291     |
|       1       |      0.86     |    0.94    |     0.90     |     262     |
|               |               |            |              |             |
| accuracy      |               |            |     0.90     |     553     |
| macro avg     |      0.90     |    0.90    |     0.90     |     553     |
| weighted avg  |      0.90     |    0.90    |     0.90     |     553     |

### Random Forest Classifier


*RandomForestClassifier* adalah algoritma pembelajaran mesin yang termasuk dalam kelompok algoritma ensambel, yang menggabungkan beberapa model pohon keputusan (*decision trees*) yang bekerja bersama-sama untuk menghasilkan prediksi akhir. *Random Forest* memperbaiki kecenderungan pohon keputusan untuk *overfitting* pada dataset pelatihan.[4] Ini adalah salah satu algoritma klasifikasi yang populer dalam pembelajaran mesin dan sering digunakan untuk tugas-tugas seperti klasifikasi teks, analisis citra, dan analisis data.

Random forestselajaran ensemble untuk klasifikasi, regresi, dan tugas lainnya yang beroperasi dengan membangun banyak pohon keputusan pada saat pelatihan. Untuk tugas klasifikasi, output dari hutan acak adalah kelas yang dipilih oleh sebagian besar pohon. Untuk tugas regresi, prediksi rata-rata atau rerata dari pohon-pohon individu dikembalikan. Hutan keputusan acak memperbaiki kecenderungan pohon keputusan untuk overfitting pada set pelatihan mereka. Secara umum, hutan acak biasanya lebih unggul daripada pohon keputusan, tetapi akurasinya lebih rendah daripada pohon yang ditingkatkan dengan gradien. Namun, karakteristik data dapat mempengaruhi kinerja mereka.

Pohon keputusan adalah struktur hierarkis yang menggunakan serangkaian keputusan kondisional untuk mengklasifikasikan atau memprediksi data. Pada setiap tingkat pohon keputusan, algoritma memilih fitur terbaik untuk membagi data berdasarkan pengukuran kualitas pemisahan seperti indeks Gini atau entropi.

*RandomForestClassifier* bekerja dengan cara membangun sejumlah pohon keputusan yang independen dan kemudian menggabungkan prediksi dari setiap pohon tersebut untuk menghasilkan prediksi akhir. Penting untuk dicatat bahwa setiap pohon keputusan dalam *RandomForestClassifier* dibangun dengan beberapa perbedaan dalam data pelatihan, yang diperkenalkan melalui teknik *bootstrap* dan pemilihan acak fitur.

Pada model *RandomForestClassifier* parameter-parameter yang digunakan adalah:
* *n_estimators* : 100
* *criterion : entropy*
* *random_state* : 42

Setelah mengatur parameter model *RandomForestClassifier*, model dilatih dengan memanggil metode 'fit' menggunakan data latih. Setelah model dilatih, dilakukan prediksi pada data uji dengan memanggil metode '*predict*'. Akurasi hasil prediksi dihitung menggunakan '*accuracy_score*' dengan membandingkan label sebenarnya (y_test) dengan prediksi (y_pred). setelah dilakukan prediksi pada data uji, langkah selanjutnya adalah mengevaluasinya menggunakan *'cross_val_score'* dengan parameter cv=10.

Tabel 5. Hasil prediksi untuk model *Random Forest Classifier*.

|               |   precision   |   recall   |   f1-score   |   support   |
|---------------|---------------|------------|--------------|-------------|
|       0       |      1.00     |    0.92    |     0.96     |     291     |
|       1       |      0.92     |    1.00    |     0.96     |     262     |
|               |               |            |              |             |
| accuracy      |               |            |     0.96     |     553     |
| macro avg     |      0.96     |    0.96    |     0.96     |     553     |
| weighted avg  |      0.96     |    0.96    |     0.96     |     553     |

### XGB Classifier

*XGBClassifier* adalah singkatan dari *eXtreme Gradient Boosting Classifier*. Ini adalah sebuah algoritma klasifikasi yang menggunakan metode *ensemble learning*, yang menggabungkan banyak pohon keputusan (*decision trees*) kecil untuk membuat prediksi akhir.

*XGBClassifier* didasarkan pada teknik yang disebut *Gradient Boosting*. *Gradient Boosting* adalah teknik di mana model prediktif dibangun secara iteratif, dengan setiap iterasi memperbaiki kelemahan model sebelumnya. XGBClassifier secara khusus menggunakan algoritma optimasi gradien stokastik untuk meningkatkan performa model secara signifikan.

Salah satu keunggulan utama *XGBClassifier* adalah kemampuannya dalam menangani dataset yang besar dengan fitur-fitur yang kompleks. Ini dapat memproses data dengan efisien dan memiliki penanganan yang baik terhadap *overfitting*. *XGBClassifier* juga mampu menangani fitur numerik dan kategorikal dengan baik, serta memberikan penanganan yang baik terhadap data yang hilang atau kosong.

*XGBClassifier* memiliki beberapa parameter yang dapat disesuaikan untuk mengoptimalkan kinerja model, seperti jumlah pohon keputusan yang digunakan, kedalaman maksimum setiap pohon, serta tingkat pembelajaran (*learning rate*). Selain itu, *XGBClassifier* juga menyediakan fitur-fitur seperti regularisasi, penanganan data yang tidak seimbang (*imbalanced data*), dan evaluasi performa yang mendalam.

Ketika *XGBClassifier* dilatih dengan data pelatihan, ia akan membangun serangkaian pohon keputusan berdasarkan gradien penurunan fungsi kerugian. Setiap pohon berkontribusi pada prediksi akhir dengan bobot yang dihitung berdasarkan kesalahan yang diperoleh oleh pohon sebelumnya. Dengan demikian, *XGBClassifier* menggabungkan prediksi dari banyak pohon keputusan untuk memberikan hasil akhir.

Secara keseluruhan, *XGBClassifier* adalah algoritma klasifikasi yang kuat dan efektif yang dapat digunakan untuk berbagai tugas pemodelan prediktif. Ini telah terbukti sukses dalam banyak kompetisi data dan digunakan secara luas dalam industri untuk masalah klasifikasi.

Pada model *XGBClassifier* parameter-parameter yang digunakan adalah:
* *max_depth* : 5
* *learning_rate* : 0.1
* *n_estimators* : 100
* *subsample* : 0.8
* *colsample_bytree* : 0.8
* *reg_alpha* : 0.1
* *reg_lambda* : 0.1

Setelah mengatur parameter model *XGBClassifier*, model dilatih dengan memanggil metode *'fit'* menggunakan data latih. Setelah model dilatih, dilakukan prediksi pada data uji dengan memanggil metode '*predict*'. Akurasi hasil prediksi dihitung menggunakan '*accuracy_score*' dengan membandingkan label sebenarnya (*y_test*) dengan prediksi (*y_pred*). setelah dilakukan prediksi pada data uji, langkah selanjutnya adalah mengevaluasinya menggunakan *'cross_val_score'* dengan parameter cv=10.


Tabel 6. Hasil prediksi untuk model *XGB Classifier*.

|               |   precision   |   recall   |   f1-score   |   support   |
|---------------|---------------|------------|--------------|-------------|
|       0       |      1.00     |    0.90    |     0.95     |     291     |
|       1       |      0.90     |    1.00    |     0.95     |     262     |
|               |               |            |              |             |
| accuracy      |               |            |     0.95     |     553     |
| macro avg     |      0.95     |    0.95    |     0.95     |     553     |
| weighted avg  |      0.95     |    0.95    |     0.95     |     553     |

### Bagging Classifier

*Bagging Classifier* juga disebut *Bootstrap aggregating*, adalah meta-algoritma *ensemble* dalam *machine learning* yang dirancang untuk meningkatkan stabilitas dan akurasi dari algoritma *machine learning* yang digunakan dalam klasifikasi statistik. Hal ini juga mengurangi varian dan membantu menghindari overfitting. Meskipun biasanya diterapkan pada metode pohon keputusan (*decision trees*), *bagging* dapat digunakan dengan metode apa pun. *Bagging* adalah kasus khusus dari pendekatan rata-rata model. Agregasi *bootstrap* dapat terkait dengan distribusi prediksi posterior.[5]

Dalam *bagging classifier*, data pelatihan yang ada dibagi menjadi beberapa subset yang diambil secara acak dengan penggantian. Setiap subset kemudian digunakan untuk melatih pengklasifikasi dasar yang sama, misalnya *decision tree* atau *k-nearest neighbors*. Setelah pengklasifikasi dasar dilatih, *bagging classifier* menggabungkan hasil presiksi dari setiap pengklasifikasi dasar menggunakan voting, dimana label yang paling sering muncul menjadi prediksi akhir.

Dalam proyek ini kami menggunakan *DecisionTreeClassifier* sebagai pengklasifikasi dsasar dalam *BaggingClassifier*. berikut parameter-parameter yang digunakan:

* *n_estimators*: 10
* *max_samples* : 0.8
* *max_features* : 0.8
* *bootstrap* : *True*
* *random_state* : 42

Setelah mengatur parameter model *BaggingClassifier*, model dilatih dengan memanggil metode *'fit'* menggunakan data latih. Setelah model dilatih, dilakukan prediksi pada data uji dengan memanggil metode '*predict*'. Akurasi hasil prediksi dihitung menggunakan '*accuracy_score*' dengan membandingkan label sebenarnya (y_test) dengan prediksi (y_pred). setelah dilakukan prediksi pada data uji, langkah selanjutnya adalah mengevaluasinya menggunakan *'cross_val_score'* dengan parameter cv=10.

Tabel 7. Hasil prediksi untuk model *Bagging Classifier*.

|               |   precision   |   recall   |   f1-score   |   support   |
|---------------|---------------|------------|--------------|-------------|
|       0       |      0.87     |    0.80    |     0.84     |     291     |
|       1       |      0.80     |    0.87    |     0.83     |     262     |
|               |               |            |              |             |
| accuracy      |               |            |     0.83     |     553     |
| macro avg     |      0.83     |    0.84    |     0.83     |     553     |
| weighted avg  |      0.84     |    0.83    |     0.83     |     553     |

### Final Report

Setelah melalui berbagai tahapan evaluasi diputuskan bahwa model terbaik yang akan digunakan adalah *ExtraTreesClassifier* sesuai dengan perhitungan metrik yang telah dijabarkan diatas. Berikut hasil evaluasi akhir dari 5 Model algoritma pada proyek kali ini.

Tabel 8. Hasil evaluasi akurasi dari masing model algoritma dengan cv=10.

|    | Nama Model             |   Akurasi |
|----|------------------------|-----------|
|  1 | ExtraTreesClassifier   |   98.0995 |
|  2 | RandomForestClassifier |   96.7431 |
|  3 | XGBClassifier          |   95.0699 |
|  4 | DecisionTreeClassifier |   88.6476 |
|  5 | BaggingClassifier      |   81.1838 |

Setelah melakukan evaluasi untuk data tes, akurasi yang dihasilkan menunjukkan bahwa model *ExtraTreesClassifier* masih menjadi yang terbaik, walaupun masih ada perbedaan dari hasil prediksinya, tetapi angka tersebut bisa menjadi patokan bagi pecinta anggur merah dan juga industri untuk menggunakan model tersebut dalam memprediksi anggur yang sesuai.

## Kesimpulan

Dari semua proses analisa prediksi pada dataset *'Red Wine Quality'* yang dilakukan Sampailah kita pada kesimpulan. Kesimpulan yang dapat dimbil untuk menjawab tujuan dari proyek ini yaitu :
- Untuk mendapatpakan hasil prediksi yang baik, pada dataset *Red Wine Quality* perlu dilakukan proses *undersampling* dan *oversampling* untuk mendapatkan rasio dataset yang seimbang dengan memanfaatkan salah satu pustaka python bernama *imblearn*.
- Model algoritma yang didapat dari *lazyPredict* adalah:
     1. *ExtraTreesClassifier*.
     2. *RandomForestClassifier*.
     3. *XGBClassifier*.
     4. *DecisionTreeClassifier*.
     5. *BaggingClassifier*.
- Didapat hasil evaluasi dari model 5 algoritma terbaik adalah sebagai berikut:
     1. *ExtraTreesClassifier* = 98.0995 %
     2. *RandomForestClassifier* = 96.7431 %
     3. *XGBClassifier* = 95-0699 %
     4. *DecisionTreeClassifier* = 88.6476 %
     5. *BaggingClassifier* = 81.1838 %  
- Dari kesebelas fitur yang terdapat dalam kandungan anggur merah, ada faktor-faktor yang memengaruhi kualitas anggur merah adalah sebagai berikut :
     1. **Chlorides**
     2. **Citric Acid**
     3. **Volatile Acidity**
     4. **Sulphates**
     5. **Alcohol**
- Untuk mendapatkan hasil kualitas yang anggur merah yang sangat bagus haruslah memperhatikan kandungan kimia seperti yang ditunjukkan pada kesimpulan diatas, namun harga yang harus dibayarpun akan lebih mahal pula dan hanya sebagian kalangan saja yang dapat membelinya. Apabila suatu perusahaan menginginkan penjualan anggur merah tetap stabil namun konsumen masih dapat menjangkau harga anggur merah, maka perlu diperhatikan juga kandungan *Sulfur Dioxide* dalam anggur merah.


## Daftar Referensi

[1] Jackson, R. S. (2014). Wine Science: Principles and Applications (4th ed.). Academic Press.

[2] P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

[3] H. He and E.A. Garcia. "Learning from imbalanced data." IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263-1284, 2009.

[4] Hastie, Trevor; Tibshirani, Robert; Friedman, Jerome (2008). The Elements of Statistical Learning (2nd ed.). Springer. ISBN 0-387-95284-5. 587-588.

[5] "The bootstrap predictive distribution is considered to be an approximation of the Bayesian predictive distribution". Bayesian bootstrap prediction, Tadayoshi Fushiki, http://dx.doi.org/10.1016/j.jspi.2009.06.007


