<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&pause=1000&color=00C853&center=true&vCenter=true&width=435&lines=Tomato+Disease+%7C+10+Classes+%7C+MobileNetV2" alt="Typing SVG" />

# 🌿 Plant Disease Detection AI
### Transfer Learning · MobileNetV2 · TensorFlow · Real-time Webcam

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge)](LICENSE)

> **Sistem deteksi penyakit daun tomat berbasis Deep Learning** yang mampu mengidentifikasi **10 kelas penyakit** secara real-time melalui webcam maupun analisis gambar statis — dibangun menggunakan arsitektur Transfer Learning MobileNetV2 dengan akurasi tinggi.

</div>

---

## 📸 Demo

| Mode File Gambar | Mode Realtime Webcam |
|:---:|:---:|
| Input gambar → Top-3 prediksi + bar chart | Live feed → overlay UI + instant diagnosis |
| `python predict.py` → pilih `1` | `python predict.py` → pilih `2` |

---

## 🎯 Highlights

- 🔬 **10 Kelas Penyakit** — mencakup bacterial spot, early/late blight, leaf mold, mosaic virus, dan lainnya  
- ⚡ **Dual Mode** — analisis dari file gambar statis atau webcam real-time  
- 🎨 **UI Overlay Profesional** — glass panel, scanline animation, corner bracket, color-coded per penyakit  
- 📊 **Top-3 Confidence** — menampilkan 3 kandidat prediksi sekaligus dengan bar chart  
- 💊 **Rekomendasi Penanganan** — setiap kelas dilengkapi info tingkat keparahan & saran treatment  
- 🧠 **Two-Stage Training** — frozen head → fine-tune untuk konvergensi optimal  

---

## 🏗️ Arsitektur Model

```
Input (224×224×3)
        │
   MobileNetV2          ← pretrained ImageNet, frozen di tahap 1
  (Feature Extractor)   ← fine-tune 30 layer terakhir di tahap 2
        │
  GlobalAveragePooling2D
        │
  BatchNormalization
        │
  Dense(512, relu)
        │
  Dropout(0.4)
        │
  Dense(256, relu)
        │
  Dropout(0.3)
        │
  Dense(10, softmax)    ← 10 kelas penyakit tomat
        │
     Output
```

**Strategi Training 2 Tahap:**

| Tahap | Layer yang Dilatih | Learning Rate | Epochs |
|:---:|---|:---:|:---:|
| **1 — Frozen** | Hanya classification head | `1e-3` | 15 |
| **2 — Fine-tune** | Head + 30 layer terakhir base | `1e-4` | 10 |

---

## 🦠 Kelas Penyakit yang Dideteksi

| # | Nama Penyakit | ID Bahasa Indonesia | Tingkat Keparahan |
|:---:|---|---|:---:|
| 1 | Bacterial Spot | Bercak Bakteri | 🟡 Sedang |
| 2 | Early Blight | Hawar Awal | 🟡 Sedang |
| 3 | Late Blight | Hawar Akhir | 🔴 Tinggi |
| 4 | Leaf Mold | Jamur Daun | 🟢 Rendah |
| 5 | Septoria Leaf Spot | Bercak Septoria | 🟡 Sedang |
| 6 | Spider Mites | Tungau Laba-laba | 🟡 Sedang |
| 7 | Target Spot | Bercak Target | 🟡 Rendah-Sedang |
| 8 | Mosaic Virus | Virus Mosaik | 🔴 Tinggi |
| 9 | Yellow Leaf Curl Virus | Virus Keriting Daun Kuning | 🔴 Tinggi |
| 10 | Healthy | Sehat | ✅ Tidak ada |

---

## 📊 Hasil Training

| Metrik | Nilai |
|---|:---:|
| Validation Accuracy | **~95%+** |
| Top-3 Accuracy | **~99%** |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |

```
Training History:
  Epoch 1/15  — val_accuracy: 0.72  ████████░░░░░░░░░░░░
  Epoch 5/15  — val_accuracy: 0.87  █████████████░░░░░░░
  Epoch 10/15 — val_accuracy: 0.93  ██████████████████░░
  Epoch 15/15 — val_accuracy: 0.95  ████████████████████  ← best checkpoint

  Fine-tune Epoch 1/10  — val_accuracy: 0.95  ████████████████████
  Fine-tune Epoch 5/10  — val_accuracy: 0.97  ████████████████████
```

> Grafik lengkap training history dan confusion matrix tersimpan di folder `models/` setelah training selesai.

---

## 🗂️ Struktur Project

```
plant-disease-ai/
│
├── 📄 train.py                  # Script training + evaluasi model
├── 📄 predict.py                # Script prediksi (file & webcam)
│
├── 📊 training_history.png      # Grafik akurasi & loss per epoch
├── 📊 confusion_matrix.png      # Confusion matrix semua kelas
│
├── 📁 models/                   # Output training (tidak diupload)
│   ├── plant_disease_final.keras
│   ├── best_model.keras
│   ├── best_model_finetuned.keras
│   └── class_names.json
│
└── 📁 dataset/                  # PlantVillage dataset (tidak diupload)
    └── PlantVillage/
        ├── Tomato_Bacterial_spot/
        ├── Tomato_Early_blight/
        └── ...
```

---

## ⚙️ Instalasi & Cara Pakai

### 1. Clone Repository
```bash
git clone https://github.com/arill2/A.I-plant-desease-detect.git
cd A.I-plant-desease-detect
```

### 2. Install Dependencies
```bash
pip install tensorflow opencv-python matplotlib scikit-learn seaborn
```

### 3. Siapkan Dataset
Download dataset PlantVillage dari Kaggle dan ekstrak ke:
```
D:\ALL PROJECT A.I\dataset\PlantVillage\
```
> 🔗 https://www.kaggle.com/datasets/emmarex/plantdisease

### 4. Training Model
```bash
python train.py
```
Proses ini akan:
- Melatih model 2 tahap (frozen → fine-tune)
- Menyimpan best checkpoint otomatis
- Generate grafik training history & confusion matrix
- Menyimpan model final ke `models/plant_disease_final.keras`

### 5. Jalankan Prediksi
```bash
python predict.py
```
```
==================================================
  PLANT DISEASE PREDICTOR
==================================================
  1. Predict dari file gambar
  2. Realtime webcam
==================================================

Pilih mode (1/2): _
```

**Mode 1 — File Gambar:**  
Masukkan path gambar daun → sistem menampilkan Top-3 prediksi + visualisasi matplotlib

**Mode 2 — Webcam Real-time:**  
Arahkan daun ke dalam kotak tengah → tekan `SPACE` untuk analisis → hasil muncul sebagai overlay UI

---

## 🛠️ Tech Stack

| Teknologi | Kegunaan |
|---|---|
| **TensorFlow / Keras** | Training & inference model deep learning |
| **MobileNetV2** | Pretrained CNN backbone (ImageNet) |
| **OpenCV** | Webcam capture, frame processing, UI overlay |
| **NumPy** | Array manipulation & preprocessing |
| **Matplotlib** | Visualisasi hasil prediksi & training history |
| **Scikit-learn** | Classification report & confusion matrix |
| **Seaborn** | Heatmap confusion matrix |

---

## 💡 Konsep yang Dipelajari

- ✅ Transfer Learning dengan pretrained CNN
- ✅ Two-stage training strategy (frozen → fine-tune)
- ✅ Data Augmentation untuk mengatasi overfitting
- ✅ Callback: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- ✅ Top-K Accuracy sebagai metrik evaluasi
- ✅ Real-time inference dengan OpenCV
- ✅ Custom UI overlay pada video feed

---

## ⚠️ Catatan

- File model `.keras` tidak diupload karena ukurannya besar (>100MB)
- Jalankan `train.py` terlebih dahulu untuk menghasilkan file model
- GPU sangat disarankan untuk proses training (training di CPU akan jauh lebih lambat)
- Pastikan path dataset dan model di `train.py` / `predict.py` sudah disesuaikan

---

<div align="center">

**Dibuat dengan ❤️ sebagai proyek pembelajaran Deep Learning**

*Jika project ini bermanfaat, jangan lupa beri ⭐ pada repository ini!*

</div>
