"""
Konfigurasi terpusat proyek Plant Disease Detection.

Semua konstanta yang dipakai bersama oleh train.py dan predict.py ada di sini.
Tujuannya sederhana: dulu jumlah kelas tertulis di tiga tempat dengan tiga angka
berbeda (15 di script training, 10 di README, 38 di docstring). Sekarang cuma
ada satu sumber kebenaran, dan daftar kelas asli dibaca dari isi folder dataset.

Modul ini SENGAJA tidak meng-import TensorFlow / OpenCV / scikit-learn,
supaya bisa diimpor dan dites tanpa memasang GPU stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- versi format
# Dinaikkan setiap kali arti file training_meta.json berubah.
FORMAT_VERSION = 2

# ---------------------------------------------------------------- path
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "PlantVillage"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "plant_disease.keras"
META_PATH = MODEL_DIR / "training_meta.json"

# ---------------------------------------------------------------- training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Split tiga arah. Dulu hanya ada train/validasi, dan set validasi itu juga
# dipakai ModelCheckpoint + EarlyStopping, jadi angka akurasinya bias seleksi.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 10
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
UNFREEZE_LAYERS = 30

# "mobilenet_v2" -> skala [-1, 1], sesuai bobot ImageNet yang dipakai MobileNetV2
# "rescale"      -> skala [0, 1], dipakai model lama di models/legacy_v1
PREPROCESS = "mobilenet_v2"

# Di bawah ambang ini model dianggap tidak yakin dan sebaiknya tidak menjawab.
# Catatan jujur: ini BUKAN deteksi out-of-distribution yang benar. Softmax bisa
# tetap tinggi untuk gambar di luar 15 kelas ini. Lihat bagian Keterbatasan di README.
CONFIDENCE_THRESHOLD = 0.60


# ---------------------------------------------------------------- profil penyakit
@dataclass(frozen=True)
class DiseaseProfile:
    name_en: str
    name_id: str
    agent: str  # Bakteri / Jamur / Virus / Hama / "-"
    severity: str  # Rendah / Sedang / Tinggi / Tidak ada
    treatment: str
    color_bgr: tuple[int, int, int]


# Kunci dictionary = nama folder persis seperti di dataset PlantVillage
# (emmarex/plantdisease di Kaggle). Nama folder itu memang tidak konsisten:
# ada yang satu underscore, ada yang dua atau tiga. Jangan "dirapikan",
# karena kunci ini harus sama persis dengan nama folder di disk.
DISEASE_INFO: dict[str, DiseaseProfile] = {
    "Pepper__bell___Bacterial_spot": DiseaseProfile(
        name_en="Bell Pepper - Bacterial Spot",
        name_id="Bercak Bakteri (Paprika)",
        agent="Bakteri",
        severity="Sedang",
        treatment=(
            "Semprot bakterisida berbasis tembaga. Hindari percikan air dari atas "
            "dan jangan bekerja saat daun masih basah, karena bakteri menyebar "
            "lewat cipratan. Rotasi tanaman minimal dua tahun."
        ),
        color_bgr=(0, 100, 255),
    ),
    "Pepper__bell___healthy": DiseaseProfile(
        name_en="Bell Pepper - Healthy",
        name_id="Paprika Sehat",
        agent="-",
        severity="Tidak ada",
        treatment="Tanaman sehat. Pertahankan penyiraman teratur dan sirkulasi udara yang baik.",
        color_bgr=(0, 200, 100),
    ),
    "Potato___Early_blight": DiseaseProfile(
        name_en="Potato - Early Blight",
        name_id="Hawar Awal (Kentang)",
        agent="Jamur",
        severity="Sedang",
        treatment=(
            "Disebabkan Alternaria solani. Semprot fungisida berbahan aktif "
            "mankozeb atau klorotalonil secara bergilir agar tidak resisten. "
            "Buang daun bawah yang terinfeksi dan lakukan rotasi tanaman."
        ),
        color_bgr=(0, 140, 255),
    ),
    "Potato___Late_blight": DiseaseProfile(
        name_en="Potato - Late Blight",
        name_id="Hawar Akhir (Kentang)",
        agent="Jamur",
        severity="Tinggi",
        treatment=(
            "Disebabkan Phytophthora infestans. Menyebar sangat cepat dan bisa "
            "menghabiskan satu petak dalam hitungan hari. Gunakan fungisida "
            "preventif sebelum gejala meluas, musnahkan tanaman terinfeksi, dan "
            "jangan menumpuk umbi dari lahan yang sudah sakit."
        ),
        color_bgr=(0, 50, 255),
    ),
    "Potato___healthy": DiseaseProfile(
        name_en="Potato - Healthy",
        name_id="Kentang Sehat",
        agent="-",
        severity="Tidak ada",
        treatment="Tanaman sehat. Jaga kelembapan tanah stabil dan hindari genangan.",
        color_bgr=(0, 200, 100),
    ),
    "Tomato_Bacterial_spot": DiseaseProfile(
        name_en="Tomato - Bacterial Spot",
        name_id="Bercak Bakteri (Tomat)",
        agent="Bakteri",
        severity="Sedang",
        treatment=(
            "Semprot bakterisida berbasis tembaga. Hindari penyiraman dari atas "
            "dan jangan memindahkan tanaman saat daun basah. Rotasi tanaman."
        ),
        color_bgr=(0, 100, 255),
    ),
    "Tomato_Early_blight": DiseaseProfile(
        name_en="Tomato - Early Blight",
        name_id="Hawar Awal (Tomat)",
        agent="Jamur",
        severity="Sedang",
        treatment=(
            "Gunakan fungisida berbahan aktif klorotalonil atau mankozeb. Buang "
            "daun yang terinfeksi, mulsa tanah untuk menahan cipratan spora, dan "
            "lakukan rotasi tanaman."
        ),
        color_bgr=(0, 140, 255),
    ),
    "Tomato_Late_blight": DiseaseProfile(
        name_en="Tomato - Late Blight",
        name_id="Hawar Akhir (Tomat)",
        agent="Jamur",
        severity="Tinggi",
        treatment=(
            "Segera semprot fungisida sistemik. Buang dan musnahkan bagian yang "
            "terinfeksi, jangan dikompos. Penyakit ini menular antar tanaman "
            "dalam satu petak dengan cepat."
        ),
        color_bgr=(0, 50, 255),
    ),
    "Tomato_Leaf_Mold": DiseaseProfile(
        name_en="Tomato - Leaf Mold",
        name_id="Jamur Daun (Tomat)",
        agent="Jamur",
        severity="Rendah",
        treatment=(
            "Perbaiki sirkulasi udara dan turunkan kelembapan ruang tanam. "
            "Fungisida hanya perlu kalau serangan sudah meluas."
        ),
        color_bgr=(100, 180, 255),
    ),
    "Tomato_Septoria_leaf_spot": DiseaseProfile(
        name_en="Tomato - Septoria Leaf Spot",
        name_id="Bercak Septoria (Tomat)",
        agent="Jamur",
        severity="Sedang",
        treatment=(
            "Buang daun bagian bawah yang terinfeksi lebih dulu. Gunakan "
            "fungisida klorotalonil secara rutin dan hindari menyiram dari atas."
        ),
        color_bgr=(0, 120, 220),
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": DiseaseProfile(
        name_en="Tomato - Two-spotted Spider Mite",
        name_id="Tungau Laba-laba (Tomat)",
        agent="Hama",
        severity="Sedang",
        treatment=(
            "Ini hama, bukan penyakit. Semprotkan air bertekanan ke sisi bawah "
            "daun untuk menekan populasi, lalu gunakan mitisida berbahan aktif "
            "abamektin bila perlu. Hindari insektisida berspektrum luas karena "
            "membunuh predator alaminya."
        ),
        color_bgr=(0, 160, 200),
    ),
    "Tomato__Target_Spot": DiseaseProfile(
        name_en="Tomato - Target Spot",
        name_id="Bercak Target (Tomat)",
        agent="Jamur",
        severity="Sedang",
        treatment=(
            "Fungisida berbahan aktif klorotalonil. Perbaiki drainase tanah dan "
            "jangan menanam terlalu rapat."
        ),
        color_bgr=(80, 150, 255),
    ),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": DiseaseProfile(
        name_en="Tomato - Yellow Leaf Curl Virus",
        name_id="Virus Keriting Daun Kuning",
        agent="Virus",
        severity="Tinggi",
        treatment=(
            "Tidak ada obat untuk tanaman yang sudah terinfeksi. Kendalikan kutu "
            "kebul (Bemisia tabaci) sebagai vektor dengan perangkap kuning dan "
            "insektisida, lalu cabut tanaman yang gejala berat agar tidak jadi "
            "sumber penularan."
        ),
        color_bgr=(0, 200, 255),
    ),
    "Tomato__Tomato_mosaic_virus": DiseaseProfile(
        name_en="Tomato - Mosaic Virus",
        name_id="Virus Mosaik (Tomat)",
        agent="Virus",
        severity="Tinggi",
        treatment=(
            "Tidak ada obat. Cabut dan musnahkan tanaman terinfeksi. Virus ini "
            "menular lewat alat dan tangan yang terkontaminasi, jadi sterilkan "
            "gunting dan cuci tangan sebelum menangani tanaman sehat."
        ),
        color_bgr=(0, 80, 200),
    ),
    "Tomato_healthy": DiseaseProfile(
        name_en="Tomato - Healthy",
        name_id="Tomat Sehat",
        agent="-",
        severity="Tidak ada",
        treatment="Tanaman sehat. Pertahankan perawatan rutin dan pemantauan berkala.",
        color_bgr=(0, 200, 100),
    ),
}


# Daftar 15 kelas yang memang ada di dataset emmarex/plantdisease.
# Dipakai untuk pengujian konsistensi, bukan untuk memaksa isi dataset.
EXPECTED_CLASSES: tuple[str, ...] = tuple(DISEASE_INFO.keys())


def profile_for(class_name: str) -> DiseaseProfile:
    """Ambil profil penyakit tanpa pernah melempar KeyError.

    Dulu, kalau model memprediksi kelas yang tidak ada di DISEASE_INFO, UI
    menampilkan "Penanganan: Tidak ada" secara diam-diam. Sekarang kita selalu
    mengembalikan profil fallback yang jujur mengatakannya.
    """
    known = DISEASE_INFO.get(class_name)
    if known is not None:
        return known
    return DiseaseProfile(
        name_en=class_name,
        name_id=class_name.replace("_", " ").strip(),
        agent="-",
        severity="Tidak diketahui",
        treatment=(
            "Kelas ini belum punya profil penanganan. Tambahkan entri di "
            "config.DISEASE_INFO sebelum dipakai di lapangan."
        ),
        color_bgr=(200, 200, 200),
    )
