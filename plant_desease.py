"""
======================================================
  Plant Disease Detection - Real-Time Camera
  Model  : MobileNetV2 Transfer Learning
  Framework : TensorFlow / Keras
  Dataset: emmarex/plantdisease (Kaggle)
======================================================

CARA PAKAI:
  1. Install dependencies:
     pip install tensorflow opencv-python matplotlib numpy pillow

  2. Ekstrak dataset Kaggle ke folder yang sama dengan script ini.
     Pastikan strukturnya:
     PlantVillage/
       ├── Tomato_Early_blight/
       ├── Tomato_Late_blight/
       ├── Potato_healthy/
       └── ... (38 kelas)

  3. Training model (lakukan sekali):
     python plant_disease.py --mode train

  4. Jalankan kamera real-time:
     python plant_disease.py --mode camera

  5. Prediksi dari 1 gambar:
     python plant_disease.py --mode predict --image foto_daun.jpg
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ============================================================
# KONFIGURASI — sesuaikan jika perlu
# ============================================================

DATASET_DIR   = "PlantVillage"         # folder hasil ekstrak zip
IMG_SIZE      = (224, 224)             # input size MobileNetV2
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-4
MODEL_PATH    = "plant_disease_model.h5"
CLASS_FILE    = "class_names.txt"


# ============================================================
# 1. PERSIAPAN DATA
# ============================================================

def prepare_data():
    """Buat data generator train & validasi dari folder dataset."""
    print("\n[1/4] Mempersiapkan dataset...")

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"\nFolder '{DATASET_DIR}' tidak ditemukan!\n"
            "Pastikan kamu sudah ekstrak plantdisease.zip dan "
            "folder PlantVillage ada di direktori yang sama dengan script ini.\n"
            "Contoh struktur:\n"
            "  D:/ALL PROJECT A.I/\n"
            "      plant_disease.py\n"
            "      PlantVillage/\n"
            "          Apple___Apple_scab/\n"
            "          Apple___Black_rot/\n"
            "          ..."
        )

    # Augmentasi untuk training agar model lebih robust
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Validasi hanya rescale
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    # Simpan nama kelas ke file
    class_names = list(train_generator.class_indices.keys())
    with open(CLASS_FILE, "w") as f:
        f.write("\n".join(class_names))

    print(f"    Total kelas  : {len(class_names)}")
    print(f"    Data training: {train_generator.samples} gambar")
    print(f"    Data validasi: {val_generator.samples} gambar")

    return train_generator, val_generator, class_names


# ============================================================
# 2. BUILD MODEL — MobileNetV2 Transfer Learning
# ============================================================

def build_model(num_classes):
    """Buat model MobileNetV2 dengan custom classifier head."""
    print("\n[2/4] Membangun model MobileNetV2...")

    # Load MobileNetV2 pretrained ImageNet, tanpa top layer
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model dulu (phase 1: feature extraction)
    base_model.trainable = False

    # Custom classifier head
    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    total     = model.count_params()
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"    Total parameter    : {total:,}")
    print(f"    Parameter trainable: {trainable:,}")

    return model, base_model


# ============================================================
# 3. TRAINING (2 Tahap)
# ============================================================

def train():
    """Training model dua tahap: feature extraction lalu fine-tuning."""
    train_gen, val_gen, class_names = prepare_data()
    num_classes = len(class_names)

    model, base_model = build_model(num_classes)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    # ── Tahap 1: Feature Extraction (base frozen) ──────────
    print("\n[3/4] Tahap 1 — Feature Extraction (10 epoch, base frozen)...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks,
    )

    # ── Tahap 2: Fine Tuning (unfreeze 50 layer terakhir) ──
    print("\n       Tahap 2 — Fine Tuning (unfreeze 50 layer terakhir)...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Gabungkan history kedua tahap
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history[key]

    print(f"\n[4/4] Model disimpan ke '{MODEL_PATH}'")
    plot_history(combined)
    return model


# ============================================================
# 4. VISUALISASI TRAINING
# ============================================================

def plot_history(history):
    """Plot grafik akurasi dan loss training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plant Disease Model — Training History", fontsize=14, fontweight="bold")

    # Akurasi
    axes[0].plot(history["accuracy"],     label="Train",      color="#2ecc71", linewidth=2)
    axes[0].plot(history["val_accuracy"], label="Validation", color="#e74c3c", linewidth=2)
    axes[0].axvline(x=9, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    axes[0].set_title("Akurasi")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history["loss"],     label="Train",      color="#2ecc71", linewidth=2)
    axes[1].plot(history["val_loss"], label="Validation", color="#e74c3c", linewidth=2)
    axes[1].axvline(x=9, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("    Grafik disimpan ke 'training_history.png'")


# ============================================================
# 5. LOAD MODEL & KELAS
# ============================================================

def load_model_and_classes():
    """Load model .h5 dan file nama kelas."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\nModel '{MODEL_PATH}' tidak ditemukan!\n"
            "Jalankan training dulu:\n"
            "  python plant_disease.py --mode train"
        )
    if not os.path.exists(CLASS_FILE):
        raise FileNotFoundError(
            f"\nFile kelas '{CLASS_FILE}' tidak ditemukan!\n"
            "Jalankan training dulu untuk generate file ini."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_FILE, "r") as f:
        class_names = f.read().splitlines()

    print(f"Model loaded: {MODEL_PATH}  ({len(class_names)} kelas)")
    return model, class_names


# ============================================================
# 6. PREDIKSI 1 GAMBAR
# ============================================================

def predict_image(image_path):
    """Prediksi penyakit dari 1 file gambar dan tampilkan hasilnya."""
    model, class_names = load_model_and_classes()

    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds    = model.predict(arr)[0]
    top3_idx = np.argsort(preds)[::-1][:3]

    print(f"\nHasil prediksi: {image_path}")
    print("-" * 50)
    for i, idx in enumerate(top3_idx):
        label = class_names[idx].replace("_", " ")
        conf  = preds[idx] * 100
        bar   = "█" * int(conf / 4)
        print(f"  #{i+1}  {label:<38} {conf:5.1f}%  {bar}")

    # Tampilkan gambar dengan hasil prediksi
    top_label = class_names[top3_idx[0]].replace("_", " ")
    top_conf  = preds[top3_idx[0]] * 100
    is_healthy = "healthy" in top_label.lower()

    plt.figure(figsize=(6, 6))
    plt.imshow(load_img(image_path))
    plt.axis("off")
    color = "green" if is_healthy else "red"
    status = "SEHAT" if is_healthy else "SAKIT"
    plt.title(f"[{status}] {top_label}\nKepercayaan: {top_conf:.1f}%",
              fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. REAL-TIME KAMERA
# ============================================================

def run_camera():
    """Buka kamera dan deteksi penyakit tanaman secara real-time."""
    model, class_names = load_model_and_classes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Kamera tidak bisa dibuka!\n"
            "Pastikan kamera terhubung dan tidak dipakai aplikasi lain."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 55)
    print("   PLANT DISEASE DETECTOR — Real-Time Camera")
    print("   Arahkan kamera ke daun atau tanaman")
    print("   [SPACE] Capture & simpan gambar")
    print("   [Q]     Keluar")
    print("=" * 55 + "\n")

    last_label = "Arahkan ke tanaman..."
    last_conf  = 0.0
    is_healthy = True
    frame_count = 0
    top5_labels = []
    top5_confs  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # ── Auto-predict setiap 15 frame (~0.5 detik) ──────
        if frame_count % 15 == 0:
            # Ambil crop area tengah frame
            crop_size = min(h, w) - 20
            cx, cy    = w // 2, h // 2
            x1 = max(cx - crop_size // 2, 0)
            y1 = max(cy - crop_size // 2, 0)
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            crop = frame[y1:y2, x1:x2]

            # Preprocess
            rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, IMG_SIZE)
            arr     = resized.astype("float32") / 255.0
            arr     = np.expand_dims(arr, axis=0)

            preds      = model.predict(arr, verbose=0)[0]
            top_idx    = np.argmax(preds)
            last_label = class_names[top_idx].replace("_", " ")
            last_conf  = preds[top_idx] * 100
            is_healthy = "healthy" in last_label.lower()

            # Top 5
            top5_idx    = np.argsort(preds)[::-1][:5]
            top5_labels = [class_names[i].replace("_", " ") for i in top5_idx]
            top5_confs  = [preds[i] * 100 for i in top5_idx]

        # ── Warna indikator ─────────────────────────────────
        color = (0, 210, 60) if is_healthy else (30, 30, 230)

        # ── Kotak scan area tengah ──────────────────────────
        box = 280
        cx, cy = w // 2, h // 2
        cv2.rectangle(frame, (cx - box, cy - box), (cx + box, cy + box), color, 3)
        cv2.putText(frame, "AREA SCAN", (cx - box + 10, cy - box + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── Panel bawah (hasil utama) ───────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        status_text = "SEHAT" if is_healthy else "TERDETEKSI PENYAKIT"
        cv2.putText(frame, status_text, (w - 320, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, last_label, (15, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {last_conf:.1f}%", (15, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # ── Panel kanan (Top 5) ─────────────────────────────
        panel_x = w - 340
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel_x - 10, 50), (w - 10, 220), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "Top 5 Prediksi:", (panel_x, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        for i, (lbl, conf) in enumerate(zip(top5_labels, top5_confs)):
            short = lbl[:30]
            bar_w = int(conf * 1.5)
            bar_color = (0, 180, 60) if i == 0 else (100, 100, 100)
            y_pos = 100 + i * 25
            cv2.rectangle(frame, (panel_x, y_pos - 12),
                          (panel_x + bar_w, y_pos + 3), bar_color, -1)
            cv2.putText(frame, f"{short} {conf:.0f}%",
                        (panel_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)

        # ── Instruksi ───────────────────────────────────────
        cv2.putText(frame, "SPACE: Capture  |  Q: Keluar",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Plant Disease Detector", frame)

        key = cv2.waitKey(1) & 0xFF

        # SPACE = capture & simpan
        if key == ord(" "):
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n  Disimpan : {filename}")
            print(f"  Hasil    : {last_label}")
            print(f"  Confidence: {last_conf:.1f}%")
            print(f"  Status   : {'SEHAT' if is_healthy else 'SAKIT'}")

        elif key == ord("q"):
            print("\nMenutup kamera...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection — MobileNetV2 + Real-Time Camera"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "camera", "predict"],
        default="camera",
        help="Mode yang dijalankan: train | camera | predict  (default: camera)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path gambar untuk mode predict (contoh: --image daun.jpg)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "camera":
        run_camera()

    elif args.mode == "predict":
        if args.image is None:
            print("Error: Tambahkan --image <path_gambar>")
            print("Contoh: python plant_disease.py --mode predict --image daun.jpg")
        else:
            predict_image(args.image)