# Plant Disease Detection — MobileNetV2 + Transfer Learning

Deteksi penyakit daun tomat, kentang, dan paprika dari gambar, dengan mode file
statis maupun webcam real-time.

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-00C853)](LICENSE)

---

## Status dan cara membaca repo ini

Repo ini pernah punya masalah konsistensi: README menyebut 10 kelas, script
training memakai 15, dan docstring menyebut 38. Angka akurasi ~95% yang dulu
ditulis di sini dihitung pada set validasi yang **juga** dipakai untuk memilih
checkpoint, jadi itu bukan estimasi yang jujur.

Sekarang repo ini sudah dirapikan:

- Jumlah kelas dibaca dari isi folder dataset, dan diverifikasi terhadap
  `config.DISEASE_INFO`. Kalau ada kelas tanpa profil penanganan, training
  berhenti di awal dengan pesan yang jelas.
- Ada tiga split: **train / validasi / test**. Angka yang dilaporkan sebagai
  hasil adalah angka **test**, karena test tidak dipakai untuk keputusan apa pun.
- Gambar duplikat dibuang sebelum split (PlantVillage banyak berisi salinan
  identik), supaya tidak bocor ke train dan test sekaligus.
- **Tidak ada angka akurasi karangan.** Angka yang ada di README ini berasal
  dari satu run nyata yang tercatat di `models/training_meta.json`, beserta
  split dan jumlah epoch-nya. Kalau kamu melatih ulang, angkanya akan berbeda.

---

## Bagaimana project ini dibangun

Bagian ini ditulis eksplisit supaya tidak ada klaim yang berlebihan.

- **Dirancang manusia:** pemilihan masalah, pemilihan pendekatan transfer
  learning, pemilihan arsitektur MobileNetV2 karena kebutuhan inferensi
  real-time di CPU, strategi dua tahap (frozen → fine-tune), skema UI, dan
  pemetaan penyakit ke tingkat keparahan beserta saran penanganan dalam Bahasa
  Indonesia.
- **Implementasi kode:** dibangun dengan bantuan AI, lalu diaudit dan diperbaiki.
- **Yang tidak diklaim:** model ini bukan hasil riset baru, dan tidak divalidasi
  pada kondisi lapangan. Lihat bagian Keterbatasan.

---

## Dataset

[PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — 15 kelas
(2 paprika, 3 kentang, 10 tomat), citra daun tunggal dengan latar seragam.

Dataset **tidak** disertakan di repo. Ekstrak sehingga strukturnya menjadi:

```
dataset/PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
├── Tomato_Septoria_leaf_spot/
├── Tomato_Spider_mites_Two_spotted_spider_mite/
├── Tomato__Target_Spot/
├── Tomato__Tomato_YellowLeaf__Curl_Virus/
├── Tomato__Tomato_mosaic_virus/
└── Tomato_healthy/
```

Nama folder memang tidak konsisten (satu, dua, atau tiga underscore). Jangan
"dirapikan", karena nama itu harus sama persis dengan kunci di
`config.DISEASE_INFO`.

---

## Arsitektur

```
Input 224×224×3  →  preprocess_input  →  skala [-1, 1]
        │
   MobileNetV2 (ImageNet)          ← tahap 1: frozen
   (feature extractor)             ← tahap 2: 30 layer terakhir dibuka
        │
  GlobalAveragePooling2D
        │
  BatchNormalization
        │
  Dense(512, relu)  →  Dropout(0.4)
        │
  Dense(256, relu)  →  Dropout(0.3)
        │
  Dense(15, softmax)
```

| Tahap | Layer dilatih | LR | Epoch |
|---|---|---|---|
| 1 — Feature extraction | head saja | `1e-3` | 10 |
| 2 — Fine-tune | head + 30 layer terakhir base | `1e-4` | 10 |

Catatan teknis: `base_model(inputs, training=False)` dipakai supaya
BatchNormalization di dalam MobileNetV2 tetap memakai statistik ImageNet selama
fine-tuning. Tanpa ini, hasil fine-tune sering justru memburuk.

---

## Instalasi

Butuh **Python 3.10–3.12** (TensorFlow belum mendukung 3.13/3.14).

```bash
git clone https://github.com/arill2/A.I-plant-desease-detect.git
cd A.I-plant-desease-detect

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Cara pakai

### 1. Cek dataset tanpa training (cepat, tidak butuh GPU)

```bash
python train.py --dry-run
```

Perintah ini hanya membaca folder, membuang duplikat, membagi split, lalu
mencetak sebaran per kelas. Pakai ini dulu sebelum menunggu training berjam-jam.

### 2. Test unit (tidak butuh TensorFlow)

```bash
python -m unittest discover -s tests -v
```

### 3. Training

```bash
python train.py
```

Output:

- `models/plant_disease.keras` — model final
- `models/training_meta.json` — kelas, preprocessing, ukuran split, metrik val & test
- `reports/training_history.png`
- `reports/test_confusion_matrix.png`
- `reports/test_classification_report.txt`

Eksperimen cepat:

```bash
python train.py --epochs-head 3 --epochs-finetune 3
```

### 4. Prediksi

```bash
python predict.py --mode list                        # daftar kelas yang didukung
python predict.py --mode file --image daun.jpg       # analisis satu gambar
python predict.py --mode webcam                      # webcam real-time
```

Di mode webcam: `SPACE` untuk analisis, `Q` untuk keluar.

### 5. Demo cepat dengan model v1 (tanpa retraining)

```bash
python predict.py --mode file --image daun.jpg \
  --model models/plant_disease_final.keras
```

Model v1 dilatih dengan preprocessing `/255`, jadi lebih lemah daripada
pipeline baru. Sidecar `models/plant_disease_final.meta.json` otomatis dipakai
supaya urutan kelas dan preprocessing tetap benar. Detail ada di
[`models/README.md`](models/README.md).

---

## Hasil run

Satu run nyata di CPU (i5-7200U, tanpa GPU), dataset penuh, 5 epoch head +
3 epoch fine-tune. Waktu total ~1 jam 50 menit.

| Metrik | Nilai |
|---|---|
| **Test accuracy** | **80,19%** |
| Test top-3 accuracy | 95,15% |
| Macro F1 (test) | 0,764 |
| Val accuracy | 80,16% |

| Data | Jumlah |
|---|---|
| Total gambar terbaca | 20.638 |
| Duplikat isi dibuang | 14 |
| Train / Val / Test | 14.434 / 3.095 / 3.095 |

**Catatan penting:** README versi lama mengklaim akurasi ~95%. Angka itu
dihitung pada set validasi yang juga dipakai memilih checkpoint, dan tidak ada
test set terpisah. Angka test yang sebenarnya adalah **80,19%**. Selisih 15
poin itu bukan kemunduran model — itu selisih antara angka yang dipilih dan
angka yang jujur.

Detail per kelas ada di [`reports/test_classification_report.txt`](reports/test_classification_report.txt)
dan [`reports/test_confusion_matrix.png`](reports/test_confusion_matrix.png).

### Tiga kelas terlemah

| Kelas | Precision | Recall | F1 |
|---|---|---|---|
| Hawar Awal (Tomat) | 0,88 | **0,15** | **0,25** |
| Bercak Target (Tomat) | 0,70 | 0,64 | 0,67 |
| Virus Mosaik (Tomat) | 0,53 | 1,00 | 0,69 |

### Dua temuan dari run ini

**1. Ada "kelas magnet".** `Tomato_Septoria_leaf_spot` punya recall 0,88 tapi
precision hanya 0,60. Ia menyerap hampir semua lesi berupa bercak: 34 dari 150
gambar Hawar Awal Tomat, 29 dari Bercak Bakteri Tomat, bahkan 41 dari Hawar
Awal Kentang. Secara visual ini masuk akal — bercak cokelat di daun memang
mirip satu sama lain pada resolusi 224×224. Konsekuensinya, akurasi total
terlihat wajar padahal satu kelas praktis tidak terdeteksi.

**2. Fine-tuning sempat merusak model.** Val accuracy sebelum fine-tune 0,749,
lalu **turun ke 0,573** di epoch fine-tune pertama, baru pulih ke 0,802 di
epoch ketiga. Penyebab paling mungkin: learning rate `1e-4` terlalu tinggi
untuk 2,3 juta parameter yang baru dibuka. Eksperimen berikutnya: turunkan ke
`1e-5`, atau bekukan BatchNormalization di base secara eksplisit.

Angka-angka ini sengaja ditulis apa adanya. Kalau ditanya "akurasinya berapa",
jawaban yang benar adalah 80,19% pada test set PlantVillage — bukan 95%.


---

## Struktur project

```
.
├── config.py                 # satu sumber kebenaran: path, kelas, profil penyakit
├── data.py                   # penemuan kelas, dedupe, split 3 arah, tf.data
├── model.py                  # MobileNetV2 + head, augmentasi, fine-tune
├── train.py                  # training 2 tahap + evaluasi test set
├── predict.py                # prediksi file & webcam
├── tests/test_data.py        # unit test tanpa TensorFlow
├── models/
│   ├── class_names.json      # 15 kelas hasil run v1 (bukti daftar kelas asli)
│   ├── plant_disease_final.meta.json
│   ├── training_meta.json    # hasil run terakhir: split, epoch, metrik test
│   └── README.md
├── reports/
│   ├── test_confusion_matrix.png
│   ├── test_classification_report.txt
│   ├── training_history.png
│   ├── history_head.csv
│   ├── history_finetune.csv
│   └── legacy_v1/            # grafik dari run lama, disimpan sebagai arsip
├── legacy/                   # script versi lama, tidak dipakai lagi
├── requirements.txt
└── LICENSE
```

---

## Keterbatasan

Bagian ini sengaja ditulis jujur. Jangan hapus kalau kamu mempresentasikan
project ini.

1. **PlantVillage adalah benchmark yang sudah jenuh.** Akurasi di dataset ini
   mendekati 99% bahkan untuk model sederhana. Angka tinggi di sini bukan
   prestasi, melainkan baseline. Kalau akurasimu 90-an%, masalahnya bukan pada
   model, tapi pada setup eksperimen.
2. **Domain shift.** Foto PlantVillage adalah satu daun di latar seragam dengan
   cahaya laboratorium. Model mudah belajar latar belakang, bukan gejalanya.
   Mohanty dkk. (2016) melaporkan akurasi ~99% di PlantVillage yang turun
   drastis (~31%) ketika diuji pada gambar lapangan. Model di repo ini belum
   diuji sama sekali pada foto lapangan.
3. **Threshold keyakinan bukan deteksi out-of-distribution.** Softmax bisa tetap
   tinggi untuk objek yang sama sekali bukan daun. Ambang `0.60` hanya mitigasi,
   bukan solusi.
4. **Tidak ada validasi agronomis.** Saran penanganan disusun dari pengetahuan
   umum dan **bukan** hasil konsultasi dengan penyuluh pertanian. Jangan dipakai
   sebagai dasar tindakan di lahan.
5. **Satu gambar, satu daun.** Tidak ada deteksi objek atau segmentasi, jadi
   foto dengan banyak daun akan menghasilkan satu label untuk seluruh frame.
6. **Belum ada evaluasi kalibrasi.** Confidence dari softmax tidak dikalibrasi
   (temperature scaling / reliability diagram belum dilakukan).

---

## Referensi

- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for
  Image-Based Plant Disease Detection.* Frontiers in Plant Science, 7, 1419.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear
  Bottlenecks.* CVPR.
- Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on
  plant health for the development of mobile and web applications.* arXiv:1511.08060.

---

## Lisensi

MIT — lihat [LICENSE](LICENSE).
