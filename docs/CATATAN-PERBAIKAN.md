# Catatan perbaikan repo

Dokumen ini untuk kamu baca sendiri sebelum wawancara berikutnya. Isinya: apa
yang salah, kenapa diperbaiki begitu, dan jawaban yang bisa kamu pakai kalau
ditanya.

---

## 1. Temuan audit dan statusnya

| # | Temuan | Bukti | Status |
|---|---|---|---|
| 1 | Tiga angka berbeda untuk jumlah kelas | `NUM_CLASSES = 15` di script, "10 kelas" di README, "38 kelas" di docstring | **Diperbaiki** — dibaca dari folder dataset, diverifikasi terhadap `config.DISEASE_INFO` |
| 2 | 5 dari 15 kelas tidak punya profil penanganan | `DISEASE_INFO` hanya 10 entri tomat; kelas paprika & kentang jatuh ke fallback dan UI menampilkan "Penanganan: Tidak ada" | **Diperbaiki** — 15 profil lengkap, plus fallback yang jujur |
| 3 | Perintah di README menunjuk file yang tidak ada | README: `python train.py` / `python predict.py`; file asli: `train_plant_desease.py` / `predict_plant_desease.py` | **Diperbaiki** — nama file sekarang sesuai |
| 4 | Klaim "model tidak diupload" padahal 82 MB `.keras` ada di repo | `models/*.keras` tracked di git | **Didokumentasikan** — lihat §5 |
| 5 | Badge MIT License menunjuk file `LICENSE` yang tidak ada | `license: null` di GitHub API | **Diperbaiki** — file `LICENSE` ditambahkan |
| 6 | `.gitignore` berisi `tts_cache/` dari project text-to-speech lain | isi `.gitignore` lama | **Diperbaiki** |
| 7 | `*.png` di-gitignore tapi PNG tetap ter-commit | inkonsistensi | **Diperbaiki** — output ke `reports/`, aturan jelas |
| 8 | Validasi dipakai untuk memilih checkpoint **dan** untuk melaporkan hasil | `ModelCheckpoint(monitor='val_accuracy')` + `EarlyStopping` lalu `evaluate(val_generator)` | **Diperbaiki** — split tiga arah, angka test terpisah |
| 9 | Tidak ada test set | hanya `validation_split=0.2` | **Diperbaiki** |
| 10 | Gambar duplikat bisa bocor antara train dan test | tidak ada deduplikasi | **Diperbaiki** — dedupe berbasis hash isi file |
| 11 | Preprocessing tidak sesuai bobot ImageNet | `img / 255.0` menghasilkan skala [0,1]; MobileNetV2 mengharapkan [-1,1] | **Diperbaiki** — `mobilenet_v2.preprocess_input` |
| 12 | `ImageDataGenerator` sudah deprecated di Keras 3 | dipakai di dua script | **Diperbaiki** — pipeline `tf.data` + layer augmentasi |
| 13 | Tidak ada `requirements.txt` | — | **Diperbaiki** |
| 14 | Tidak ada test otomatis | — | **Diperbaiki** — `tests/test_data.py`, 18 test, tanpa butuh TensorFlow |
| 15 | Path Windows hardcoded (`D:\ALL PROJECT A.I\...`) | dua script | **Diperbaiki** — path relatif ke root repo |
| 16 | Loss `categorical_crossentropy` tapi label dari `tf.data` masih integer (rank 1) | ketahuan saat run nyata: `ValueError: target.shape=(None,), output.shape=(None, 15)` | **Diperbaiki** — `sparse_categorical_crossentropy` + `SparseCategoricalAccuracy` / `SparseTopKCategoricalAccuracy` |

---

## 2. Kenapa perubahan ini penting (versi yang bisa kamu jelaskan)

### Split tiga arah

Kalau set validasi dipakai untuk memilih checkpoint **dan** untuk melaporkan
akurasi, maka angka yang dilaporkan adalah **nilai maksimum dari banyak
percobaan**, bukan estimasi performa. Ini disebut *selection bias*. Model
kelihatan lebih baik daripada kenyataannya, dan tidak ada cara mengetahui
seberapa besar selisihnya. Solusinya cuma satu: set terpisah yang tidak pernah
dilihat selama training maupun pemilihan model.

Sekarang `train.py` mencetak dua baris:

```
[VAL ] dipakai untuk memilih checkpoint, jadi angkanya optimistis
[TEST] tidak dipakai untuk keputusan apa pun, ini angka jujurnya
```

### Deduplikasi

PlantVillage berisi banyak gambar yang benar-benar identik. Kalau salinan yang
sama jatuh di train dan test, model seolah-olah sudah "melihat" jawabannya, dan
akurasi test jadi tidak bermakna. Dedupe berbasis SHA-1 isi file memastikan
satu gambar hanya muncul di satu split.

### Preprocessing

Bobot ImageNet untuk MobileNetV2 dilatih dengan input berskala **[-1, 1]**.
Kalau kamu memberi **input [0, 1]** (efek dari `/255`), distribusinya bergeser
dari yang pernah dilihat bobot itu. Transfer learning masih jalan karena head
beradaptasi dan base di-fine-tune, tapi kamu membuang sebagian keuntungan
pretrained. Ini bukan bug yang bikin crash — ini bug yang bikin hasil lebih
buruk tanpa ada error.

### Kelas yang tidak punya profil

Ini contoh bug yang paling berbahaya: **tidak ada error, tapi outputnya salah**.
Model memprediksi `Potato___Late_blight`, `DISEASE_INFO.get()` mengembalikan
`{}`, dan UI menampilkan "Penanganan: Tidak ada" — untuk penyakit yang justru
paling mematikan di antara 15 kelas itu. Sekarang `train.py` gagal di awal kalau
ada kelas tanpa profil, dan `profile_for()` selalu mengembalikan profil fallback
yang menyatakan profilnya belum ada, bukan diam-diam kosong.

---

## 3. Pertanyaan yang kemungkinan besar ditanya, dan jawaban yang jujur

**"Berapa akurasinya?"**

> "Di test set hasil run saya X%. Tapi saya harus bilang bahwa angka itu tidak
> berarti banyak, karena PlantVillage itu benchmark yang sudah jenuh — akurasi
> 99% ke atas sudah biasa di sana. Yang lebih menarik justru kenapa angka itu
> tidak bertahan di lapangan."

**"Kenapa MobileNetV2?"**

> "Karena target saya inferensi real-time di webcam dengan CPU biasa.
> MobileNetV2 punya sekitar 2,2 juta parameter di base-nya dan itu yang paling
> masuk untuk anggaran latensi itu. Saya belum melakukan benchmark sistematis
> melawan EfficientNet-Lite atau MobileNetV3 — itu yang mau saya ukur
> berikutnya, dengan FPS dan ukuran model sebagai pembanding."

Kalau dia tanya lanjut: **jangan mengarang angka**. Bilang saja kamu belum
mengukurnya. Itu lebih kuat daripada angka palsu yang ketahuan.

**"Kenapa fine-tune 30 layer terakhir, kenapa bukan 50 atau semua?"**

> "Itu keputusan empiris, bukan hasil ablation. 30 layer terakhir memberi
> keseimbangan antara adaptasi ke domain daun dan risiko overfit, karena
> dataset-nya tidak besar. Saya belum membandingkan 10/30/50 secara sistematis,
> jadi saya tidak bisa bilang 30 itu yang optimal."

**"Bagian mana yang kamu tulis sendiri?"**

> "Saya yang menentukan masalah, pendekatan, arsitektur, dan UX-nya. Implementasi
> kode saya bangun dengan bantuan AI, lalu saya audit. Yang saya dalami betul
> adalah bagian evaluasi dan datasetnya, karena di situ letak kesalahan yang
> paling sering terjadi dan paling tidak kelihatan."

**"Kalau model ini salah di lapangan, kamu debug dari mana?"**

Urutannya:

1. Lihat confusion matrix per kelas — apakah errornya sistematis di kelas
   tertentu, atau acak?
2. Cek apakah model sebenarnya belajar latar belakang. Uji dengan gambar daun
   yang ditempel di latar berbeda. Kalau prediksinya berubah drastis, modelnya
   belajar background.
3. Cek distribusi input: pencahayaan, blur, jarak, sudut. Bandingkan dengan
   distribusi PlantVillage.
4. Cek konsistensi preprocessing antara training dan inference — ini penyebab
   paling umum model "bagus di notebook, jelek di aplikasi".
5. Kalau semuanya beres, masalahnya data: tambah data lapangan, bukan menambah
   epoch.

**"Apa keterbatasan terbesar project ini?"**

> "Dua. Pertama, domain shift: model dilatih pada foto laboratorium, dan riset
> menunjukkan performa model seperti ini jatuh drastis di foto lapangan. Kedua,
> tidak ada validasi agronomis pada saran penanganannya — itu pengetahuan umum,
> bukan hasil konsultasi dengan penyuluh."

---

## 4. Yang perlu kamu kerjakan sendiri (jangan dilewat)

Perbaikan kode tidak otomatis membuatmu paham. Ini yang harus kamu jalankan
sendiri, karena hanya kamu yang punya dataset dan hasil training-nya:

- [ ] `python -m unittest discover -s tests -v` — pastikan 18 test lulus
- [ ] `python train.py --dry-run` — lihat sebaran split dataset aslimu
- [ ] `python train.py` — training penuh, lalu **baca** `models/training_meta.json`
- [ ] Catat angka test yang sebenarnya. Jangan tulis angka val sebagai hasil.
- [ ] Lihat `reports/test_confusion_matrix.png`. Cari kelas yang paling sering
      tertukar. Pikirkan kenapa (gejala mirip? jumlah sampel sedikit?).
- [ ] Uji model pada 5 foto daun asli dari kebun/halaman, bukan dari PlantVillage.
      Ini ujian sebenarnya. Kemungkinan besar hasilnya mengecewakan — dan justru
      itu materi presentasi terbaikmu.
- [ ] Ukur FPS webcam, lalu bandingkan dengan satu arsitektur lain. Ini yang
      mengubah "saya pilih MobileNetV2" menjadi "saya pilih MobileNetV2 karena
      datanya bilang begitu".

---

## 5. Soal 82 MB model di repo

Model v1 (`best_model.keras`, `best_model_finetuned.keras`,
`plant_disease_final.keras`) masih ter-track di git. Saya **tidak** menghapusnya
dari commit ini supaya file di disk kamu tidak hilang.

Kalau mau repo lebih ringan, jalankan di komputermu setelah retraining berhasil:

```bash
# berhenti melacak, tapi file tetap ada di disk
git rm --cached models/best_model.keras
git rm --cached models/best_model_finetuned.keras
git rm --cached models/plant_disease_final.keras
git commit -m "chore: stop tracking model weights"
```

Yang tidak bisa dihapus dengan cara di atas adalah riwayatnya — blob 82 MB itu
tetap ada di commit lama. Untuk benar-benar membuangnya perlu menulis ulang
history (`git filter-repo`). Itu opsional, dan risikonya lebih besar daripada
manfaatnya untuk repo portofolio. Cukup lakukan `git rm --cached` di atas.

---

## 6. Ringkasan file

| File | Status | Keterangan |
|---|---|---|
| `config.py` | baru | satu sumber kebenaran untuk path, kelas, profil penyakit |
| `data.py` | baru | penemuan kelas, dedupe, split 3 arah, `tf.data` |
| `model.py` | baru | MobileNetV2 + head, augmentasi, fine-tune |
| `train.py` | baru | training 2 tahap + evaluasi test set |
| `predict.py` | baru | prediksi file & webcam, dengan ambang keyakinan |
| `tests/test_data.py` | baru | 18 unit test, tidak butuh TensorFlow |
| `requirements.txt` | baru | dependensi terkunci |
| `LICENSE` | baru | MIT, sesuai badge yang sudah ada di README |
| `.gitignore` | ditulis ulang | `tts_cache/` dibuang, aturan model & output jelas |
| `README.md` | ditulis ulang | angka kelas benar, tanpa angka akurasi karangan, ada bagian Keterbatasan |
| `models/README.md` | baru | bedakan artefak v1 vs v2 |
| `models/plant_disease_final.meta.json` | baru | sidecar supaya model v1 masih bisa dipakai demo |
| `legacy/*` | dipindah | script versi lama, disimpan sebagai arsip |
| `reports/legacy_v1/*` | dipindah | grafik dari run lama |
