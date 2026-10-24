# Script versi lama

Tiga script di folder ini adalah versi pertama project, disimpan sebagai arsip
dan **tidak dipakai lagi**.

| File | Asal | Kenapa ditinggalkan |
|---|---|---|
| `plant_desease.py` | v0 | Docstring menyebut 38 kelas (salah), tidak ada split test, augmentasi `ImageDataGenerator` |
| `train_plant_desease.py` | v1 | `NUM_CLASSES = 15` hardcoded, tidak ada test set, preprocessing `/255` |
| `predict_v1.py` | v1 | `DISEASE_INFO` hanya 10 kelas, 5 kelas lain jatuh ke fallback kosong |

Yang dipakai sekarang: `train.py`, `predict.py`, `config.py`, `data.py`, `model.py`
di root repo.

Untuk demo cepat tanpa retraining, model v1 masih bisa dipakai lewat:

```bash
python predict.py --mode file --image daun.jpg --model models/plant_disease_final.keras
```
