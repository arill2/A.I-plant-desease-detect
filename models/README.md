# Isi folder models/

## Artefak v1 (warisan, jangan dijadikan acuan)

| File | Keterangan |
|---|---|
| `plant_disease_final.keras` | Model hasil training v1, 15 kelas |
| `best_model.keras` | Checkpoint terbaik tahap 1 (base frozen) |
| `best_model_finetuned.keras` | Checkpoint terbaik tahap 2 (fine-tune) |
| `class_names.json` | Daftar 15 kelas hasil `flow_from_directory` pada run v1 |
| `plant_disease_final.meta.json` | Sidecar supaya model v1 masih bisa dipakai `predict.py` |

Model v1 memakai preprocessing `/255` (skala 0..1), sedangkan pipeline baru
memakai `mobilenet_v2.preprocess_input` (skala -1..1) yang memang diharapkan
bobot ImageNet. Karena itu model v1 **tidak boleh** dipakai untuk mengukur
pipeline baru.

Angka akurasi yang pernah ditulis di README lama (~95%) berasal dari set
validasi yang juga dipakai `ModelCheckpoint` dan `EarlyStopping`, jadi itu
nilai maksimum yang sudah dipilih, bukan estimasi yang jujur. Tidak ada test
set terpisah pada run v1.

### Masih ingin demo tanpa retraining?

```bash
python predict.py --mode file --image contoh_daun.jpg \
  --model models/plant_disease_final.keras
```

`predict.py` otomatis menemukan `plant_disease_final.meta.json` di sebelah
file model, jadi urutan kelas dan cara preprocessing tetap benar.

## Artefak v2 (hasil `python train.py`)

| File | Keterangan |
|---|---|
| `plant_disease.keras` | Model final pipeline baru |
| `training_meta.json` | Kelas, preprocessing, split, metrik val **dan test** |
| `best_head.keras` | Checkpoint tahap 1 |
| `best_finetuned.keras` | Checkpoint tahap 2 |

Model `.keras` sengaja masuk `.gitignore` karena berukuran puluhan MB dan bisa
dibuat ulang. Yang perlu di-commit hanya `training_meta.json` (kecil), karena
file itu yang mencatat angka test yang sebenarnya.

Setelah `train.py` selesai dan hasilnya sudah kamu cek, file v1 di atas boleh
dihapus.
