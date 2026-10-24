"""
Prediksi: dari file gambar atau webcam real-time.

Perbedaan penting dibanding versi lama:

1. Daftar kelas, ukuran input, dan cara preprocessing dibaca dari
   models/training_meta.json yang ditulis oleh train.py. Jadi model dan kode
   prediksi tidak mungkin lagi berbeda pendapat soal urutan kelas.
2. Kalau model memprediksi kelas yang tidak punya profil, UI tidak lagi
   diam-diam menampilkan "Penanganan: Tidak ada" -- ada fallback yang jujur.
3. Ada ambang keyakinan. Di bawah ambang, aplikasi bilang "tidak yakin"
   daripada memaksakan label penyakit untuk gambar yang bukan daun.

Pakai:
    python predict.py --mode list
    python predict.py --mode file --image daun.jpg
    python predict.py --mode webcam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config

# ----------------------------------------------------------------- artifact


def load_artifact(model_path: Path | None = None, meta_path: Path | None = None):
    """Muat model dan metadata sidecar-nya."""
    import tensorflow as tf

    model_path = model_path or config.MODEL_PATH

    # Kalau model yang dimuat bukan model default dan --meta tidak diberikan,
    # coba cari sidecar bernama <model>.meta.json di sebelah file model.
    if meta_path is None and model_path != config.MODEL_PATH:
        sidecar = model_path.with_suffix(".meta.json")
        if sidecar.exists():
            meta_path = sidecar
    meta_path = meta_path or config.META_PATH

    if not model_path.exists():
        raise SystemExit(
            f"Model tidak ditemukan: {model_path}\n"
            "Latih dulu: python train.py\n"
            "Kalau kamu masih punya model lama (versi /255), lihat models/legacy_v1/README.md"
        )
    if not meta_path.exists():
        raise SystemExit(
            f"Metadata tidak ditemukan: {meta_path}\n"
            "Metadata wajib ada supaya urutan kelas dan preprocessing tidak salah. "
            "Jalankan ulang train.py."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(model_path)
    return model, meta


# ----------------------------------------------------------------- prediksi


def predict_rgb(model, meta: dict, rgb_image):
    """Prediksi dari array RGB uint8, kembalikan daftar top-3 + status yakin."""
    import numpy as np
    import cv2

    img_size = tuple(meta["img_size"])
    preprocess = meta["preprocess"]
    class_names = meta["class_names"]

    resized = cv2.resize(rgb_image, img_size, interpolation=cv2.INTER_AREA)
    array = resized.astype("float32")

    if preprocess == "mobilenet_v2":
        # preprocess_input bawaan Keras: skala [0, 255] -> [-1, 1]
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        array = preprocess_input(array)
    elif preprocess == "rescale":
        array = array / 255.0
    else:
        raise ValueError(f"Mode preprocess tidak dikenal: {preprocess}")

    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    top_indices = probabilities.argsort()[::-1][:3]

    results = [
        {
            "class": class_names[index],
            "profile": config.profile_for(class_names[index]),
            "confidence": float(probabilities[index]),
        }
        for index in top_indices
    ]

    threshold = float(meta.get("confidence_threshold", config.CONFIDENCE_THRESHOLD))
    confident = results[0]["confidence"] >= threshold
    return results, confident, threshold


def load_image_rgb(path: Path):
    """Baca gambar sebagai array RGB, sekaligus membetulkan rotasi EXIF."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


# ----------------------------------------------------------------- UI gambar


def configure_matplotlib(show: bool) -> None:
    """Pilih backend sebelum pyplot diimpor.

    Di mesin tanpa display (server, CI), memaksa backend interaktif bikin crash.
    Sebaliknya, memaksa Agg saat user memang ingin melihat hasil bikin
    plt.show() tidak melakukan apa-apa.
    """
    import os
    import sys

    import matplotlib

    has_display = os.name == "nt" or sys.platform == "darwin" or bool(os.environ.get("DISPLAY"))
    if not show or not has_display:
        matplotlib.use("Agg")


def render_file_result(
    image_rgb, results, confident: bool, threshold: float, out_path: Path, total_classes: int
):
    import matplotlib.pyplot as plt

    top = results[0]
    profile = top["profile"]

    fig, (ax_image, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    ax_image.imshow(image_rgb)
    ax_image.axis("off")

    if confident:
        title = f"{profile.name_id}\n{top['confidence'] * 100:.1f}%  ({profile.agent} / {profile.severity})"
        title_color = "#2ecc71" if profile.severity == "Tidak ada" else "#ff6b6b"
    else:
        title = f"Tidak yakin ({top['confidence'] * 100:.1f}% < {threshold * 100:.0f}%)"
        title_color = "#f1c40f"

    ax_image.set_title(title, fontsize=15, color=title_color, fontweight="bold", pad=14)

    names = [item["profile"].name_id[:24] for item in results]
    values = [item["confidence"] * 100 for item in results]
    colors = ["#4a7cf7", "#6b8fd8", "#8fa8c8"]

    ax_bar.set_facecolor("#16213e")
    bars = ax_bar.barh(names, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax_bar.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
    ax_bar.set_xlim(0, 112)
    ax_bar.set_xlabel("Confidence (%)", color="white")
    ax_bar.set_title("Top-3 prediksi", color="white", fontweight="bold")
    ax_bar.tick_params(colors="white")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.invert_yaxis()

    if confident:
        footer = f"Tingkat keparahan: {profile.severity}\n\nSaran: {profile.treatment}"
    else:
        footer = (
            f"Model tidak cukup yakin. Kemungkinan objek bukan salah satu dari {total_classes} "
            "kelas yang dilatih, atau gambar kurang jelas. "
            "Jangan pakai hasil ini sebagai dasar tindakan."
        )

    fig.text(
        0.5,
        0.005,
        footer,
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
        linespacing=1.5,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#0f3460", edgecolor=title_color, alpha=0.9),
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ----------------------------------------------------------------- UI webcam


def draw_corners(frame, cx, cy, box_size, color, thickness=3, length=30):
    import cv2

    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


def draw_glass_panel(frame, x, y, width, height, title, lines, color):
    import cv2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.rectangle(frame, (x, y), (x + width, y + 35), color, -1)

    text_color = (0, 0, 0) if sum(color) > 350 else (255, 255, 255)
    cv2.putText(frame, title, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    current_y = y + 60
    for text, scale, line_color, thickness in lines:
        for part in text.split("\n"):
            cv2.putText(
                frame, part, (x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, line_color, thickness
            )
            current_y += int(scale * 35) + 5
        current_y += 5


def wrap_text(text: str, width: int = 45) -> str:
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def run_webcam(model, meta: dict) -> None:
    import cv2

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Pastikan tidak dipakai aplikasi lain.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threshold = float(meta.get("confidence_threshold", config.CONFIDENCE_THRESHOLD))
    print("\n[WEBCAM] SPACE = analisis, Q = keluar")

    last_results = None
    last_confident = False
    scan_offset, scan_direction = 0, 1

    while True:
        ok, frame = capture.read()
        if not ok:
            print("[WEBCAM] Gagal membaca frame.")
            break

        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        box_size = 320

        # Hijau saat belum ada hasil, kuning saat model ragu, warna profil saat yakin.
        if last_results is None:
            base_color = (0, 255, 128)
        elif last_confident:
            base_color = last_results[0]["profile"].color_bgr
        else:
            base_color = (0, 200, 255)

        scan_offset += 6 * scan_direction
        if scan_offset > box_size:
            scan_offset, scan_direction = box_size, -1
        elif scan_offset < 0:
            scan_offset, scan_direction = 0, 1
        scan_y = cy - box_size // 2 + scan_offset

        draw_corners(frame, cx, cy, box_size, base_color, thickness=4)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (cx - box_size // 2, cy - box_size // 2),
            (cx + box_size // 2, cy + box_size // 2),
            base_color,
            -1,
        )
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        cv2.line(frame, (cx - box_size // 2, scan_y), (cx + box_size // 2, scan_y), base_color, 2)
        cv2.circle(frame, (cx, scan_y), 4, (255, 255, 255), -1)
        cv2.putText(
            frame,
            "Arahkan daun ke dalam kotak",
            (cx - 150, cy - box_size // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            base_color,
            2,
        )

        if last_results:
            top = last_results[0]
            profile = top["profile"]
            if last_confident:
                label = profile.name_id
                confidence_text = f"Confidence: {top['confidence'] * 100:.1f}%"
                severity_text = f"Keparahan : {profile.severity}"
                treatment_text = wrap_text(profile.treatment)
            else:
                label = "Tidak yakin"
                confidence_text = f"Confidence: {top['confidence'] * 100:.1f}% (< {threshold * 100:.0f}%)"
                severity_text = "Keparahan : -"
                treatment_text = "Model ragu. Pastikan objek adalah daun dan pencahayaan cukup."

            left_panel = [
                (f"Penyakit  : {label}", 0.7, base_color, 2),
                (confidence_text, 0.6, (255, 255, 255), 1),
                (severity_text, 0.6, (0, 165, 255), 1),
                ("Penanganan:", 0.55, (200, 200, 200), 1),
                (treatment_text, 0.48, (255, 255, 255), 1),
            ]
            draw_glass_panel(frame, 20, height - 300, 520, 280, "HASIL DETEKSI", left_panel, base_color)

            right_panel = []
            for index, item in enumerate(last_results):
                right_panel.append((f"{index + 1}. {item['profile'].name_id[:20]}", 0.55, (220, 220, 220), 1))
                right_panel.append((f"   {item['confidence'] * 100:.1f}%", 0.5, (255, 255, 255), 1))
            draw_glass_panel(
                frame, width - 320, height - 300, 300, 280, "TOP 3 PREDIKSI", right_panel, (255, 100, 100)
            )

        cv2.rectangle(frame, (0, 0), (width, 50), (10, 10, 15), -1)
        cv2.putText(
            frame,
            "TEKAN SPACE UNTUK ANALISIS  |  Q UNTUK KELUAR",
            (width // 2 - 250, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Plant Disease Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            roi_bgr = frame[
                cy - box_size // 2 : cy + box_size // 2,
                cx - box_size // 2 : cx + box_size // 2,
            ]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            last_results, last_confident, _ = predict_rgb(model, meta, roi_rgb)
            top = last_results[0]
            state = "yakin" if last_confident else "TIDAK YAKIN"
            print(
                f"[PREDICT] {top['profile'].name_id} - {top['confidence'] * 100:.1f}% ({state})"
            )

    capture.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------- CLI


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi penyakit daun")
    parser.add_argument("--mode", choices=["file", "webcam", "list"], default="file")
    parser.add_argument("--image", type=Path, help="Path gambar untuk --mode file")
    parser.add_argument("--out", type=Path, default=Path("prediction_result.png"))
    parser.add_argument("--no-show", action="store_true", help="Jangan buka jendela matplotlib")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.mode == "list":
        print(f"Model artefak : {config.MODEL_PATH}")
        print(f"Metadata      : {config.META_PATH}\n")

        # Kalau pipeline baru belum pernah dilatih, tetap berguna: tampilkan
        # kelas dari sidecar model v1 yang masih tersedia.
        meta_file = config.META_PATH
        source = "v2 (models/training_meta.json)"
        if not meta_file.exists():
            legacy = config.MODEL_DIR / "plant_disease_final.meta.json"
            if legacy.exists():
                meta_file = legacy
                source = "v1 (models/plant_disease_final.meta.json) - jalankan train.py untuk versi baru"

        if not meta_file.exists():
            print("Belum ada metadata, jalankan train.py dulu.")
            return 0

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        classes = meta["class_names"]
        metrics = meta.get("metrics") or {}
        print(f"Sumber        : {source}")
        print(f"Jumlah kelas  : {len(classes)}")
        if metrics.get("test_accuracy") is not None:
            print(
                f"Test akurasi  : {metrics['test_accuracy'] * 100:.2f}% "
                f"(top-3 {metrics.get('test_top3_accuracy', 0) * 100:.2f}%)"
            )
        else:
            print("Test akurasi  : belum ada (model v1 tidak dievaluasi pada test set)")
        print()
        for index, name in enumerate(classes, start=1):
            profile = config.profile_for(name)
            print(f"  {index:>2}. {name:<48} {profile.name_id}")
        return 0

    model, meta = load_artifact(args.model, args.meta)

    if args.mode == "webcam":
        run_webcam(model, meta)
        return 0

    if args.image is None:
        raise SystemExit("--image wajib diisi untuk --mode file")
    if not args.image.exists():
        raise SystemExit(f"File tidak ditemukan: {args.image}")

    image_rgb = load_image_rgb(args.image)
    results, confident, threshold = predict_rgb(model, meta, image_rgb)

    print(f"\n[PREDICT] {args.image.name}")
    for index, item in enumerate(results, start=1):
        print(f"  {index}. {item['profile'].name_id:<28} {item['confidence'] * 100:6.2f}%")

    if confident:
        top = results[0]
        print(f"\n  Status    : {top['profile'].severity}")
        print(f"  Penanganan: {top['profile'].treatment}")
    else:
        print(
            f"\n  [TIDAK YAKIN] Confidence tertinggi {results[0]['confidence'] * 100:.1f}% "
            f"di bawah ambang {threshold * 100:.0f}%."
        )
        print("  Jangan pakai hasil ini sebagai dasar tindakan.")

    configure_matplotlib(show=not args.no_show)

    render_file_result(
        image_rgb, results, confident, threshold, args.out, total_classes=len(meta["class_names"])
    )

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.imshow(plt.imread(args.out))
        plt.axis("off")
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
