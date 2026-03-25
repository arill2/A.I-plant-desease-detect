import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH      = r"D:\ALL PROJECT A.I\models\plant_disease_final.keras"
CLASS_NAMES_PATH= r"D:\ALL PROJECT A.I\models\class_names.json"
IMG_SIZE        = (224, 224)

# Load model & class names
print("[LOAD] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)

print(f"[LOAD] Model loaded! ({len(CLASS_NAMES)} kelas)")

# ─────────────────────────────────────────────
#  DISEASE INFO
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "en": "Bacterial Spot",
        "id": "Bercak Bakteri",
        "severity": "Sedang",
        "treatment": "Semprot bakterisida berbasis tembaga. Hindari penyiraman dari atas.",
        "color": (0, 100, 255)
    },
    "Tomato_Early_blight": {
        "en": "Early Blight",
        "id": "Hawar Awal",
        "severity": "Sedang",
        "treatment": "Gunakan fungisida. Buang daun yang terinfeksi. Rotasi tanaman.",
        "color": (0, 140, 255)
    },
    "Tomato_healthy": {
        "en": "Healthy",
        "id": "Sehat",
        "severity": "Tidak ada",
        "treatment": "Tanaman sehat! Pertahankan perawatan rutin.",
        "color": (0, 200, 100)
    },
    "Tomato_Late_blight": {
        "en": "Late Blight",
        "id": "Hawar Akhir",
        "severity": "Tinggi",
        "treatment": "Segera semprot fungisida. Buang dan bakar bagian terinfeksi.",
        "color": (0, 50, 255)
    },
    "Tomato_Leaf_Mold": {
        "en": "Leaf Mold",
        "id": "Jamur Daun",
        "severity": "Rendah",
        "treatment": "Perbaiki sirkulasi udara. Kurangi kelembaban. Fungisida jika parah.",
        "color": (100, 180, 255)
    },
    "Tomato_Septoria_leaf_spot": {
        "en": "Septoria Leaf Spot",
        "id": "Bercak Septoria",
        "severity": "Sedang",
        "treatment": "Buang daun bawah yang terinfeksi. Gunakan fungisida secara rutin.",
        "color": (0, 120, 220)
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "en": "Spider Mites",
        "id": "Tungau Laba-laba",
        "severity": "Sedang",
        "treatment": "Semprotkan air kuat ke daun. Gunakan mitisida atau insektisida.",
        "color": (0, 160, 200)
    },
    "Tomato__Target_Spot": {
        "en": "Target Spot",
        "id": "Bercak Target",
        "severity": "Rendah-Sedang",
        "treatment": "Fungisida berbasis chlorothalonil. Jaga drainase tanah.",
        "color": (80, 150, 255)
    },
    "Tomato__Tomato_mosaic_virus": {
        "en": "Mosaic Virus",
        "id": "Virus Mosaik",
        "severity": "Tinggi",
        "treatment": "Tidak ada obat. Cabut dan musnahkan tanaman. Kontrol serangga vektor.",
        "color": (0, 80, 200)
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "en": "Yellow Leaf Curl Virus",
        "id": "Virus Keriting Daun Kuning",
        "severity": "Tinggi",
        "treatment": "Kontrol kutu kebul (whitefly). Cabut tanaman terinfeksi parah.",
        "color": (0, 200, 255)
    },
}

# ─────────────────────────────────────────────
#  PREDICT FUNCTION
# ─────────────────────────────────────────────
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]

    results = []
    for idx in top3_idx:
        results.append({
            "class":      CLASS_NAMES[idx],
            "confidence": float(preds[idx]),
            "info":       DISEASE_INFO.get(CLASS_NAMES[idx], {})
        })
    return results

def show_prediction(img_path, results):
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Gambar daun ──
    ax1.imshow(img_cv)
    ax1.axis('off')
    top = results[0]
    info = top.get("info", {})
    label = info.get("id", top["class"])
    conf  = top["confidence"]
    color_bgr = info.get("color", (255,255,255))
    color_mpl = tuple(c/255 for c in reversed(color_bgr))

    ax1.set_title(f"🌿 Hasil Prediksi 🌿\n{label} ({conf*100:.1f}%)",
                  fontsize=16, color=color_mpl, fontweight='bold',
                  pad=15)

    # Styling modern untuk border
    for spine in ax1.spines.values():
        spine.set_edgecolor(color_mpl)
        spine.set_linewidth(3)

    # ── Bar chart top-3 ──
    ax2.set_facecolor('#16213e')
    names  = [DISEASE_INFO.get(r["class"], {}).get("id", r["class"])[:18] for r in results]
    confs  = [r["confidence"]*100 for r in results]
    colors = [tuple(c/255 for c in reversed(
                DISEASE_INFO.get(r["class"], {}).get("color", (200,200,200))
              )) for r in results]

    bars = ax2.barh(names, confs, color=colors, edgecolor='white', linewidth=1)
    for bar, conf_val in zip(bars, confs):
        ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                 f'{conf_val:.1f}%', va='center', ha='left',
                 color='white', fontsize=11, fontweight='bold')

    ax2.set_xlim(0, 115)
    ax2.set_xlabel('Confidence (%)', color='white', fontsize=12)
    ax2.set_title('Top-3 Kemungkinan', color='white', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white', labelsize=11)
    
    # Hilangkan border yang tidak perlu
    ax2.spines['bottom'].set_color('#aaaaaa')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#aaaaaa')
    ax2.invert_yaxis()

    # ── Info box (Lebih elegan) ──
    if info:
        severity = info.get("severity", "-")
        treatment = info.get("treatment", "-")
        info_text = (
            f"⚠️ Tingkat Keparahan: {severity}\n\n"
            f"💡 Saran Penanganan:\n"
            f"{treatment}"
        )
        if severity == "Tidak ada":
            info_text = "✨ Tanaman sehat!\n\n💡 Pertahankan perawatan rutin Anda."
            
        fig.text(0.5, 0.0, info_text, ha='center', va='bottom',
                 fontsize=11, color='white', linespacing=1.5,
                 bbox=dict(boxstyle='round,pad=1', facecolor='#0f3460',
                           edgecolor=color_mpl, linewidth=2, alpha=0.9))

    plt.suptitle("✨ Plant Disease Detection — Advanced Analysis ✨",
                 fontsize=18, color='white', fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=200, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.show()
    print("[SAVED] prediction_result.png")


# ─────────────────────────────────────────────
#  DRAWING UTILS
# ─────────────────────────────────────────────
def draw_corners(frame, cx, cy, box_size, color, thick=3, length=30):
    x1, y1 = cx - box_size//2, cy - box_size//2
    x2, y2 = cx + box_size//2, cy + box_size//2
    # TL
    cv2.line(frame, (x1, y1), (x1+length, y1), color, thick)
    cv2.line(frame, (x1, y1), (x1, y1+length), color, thick)
    # TR
    cv2.line(frame, (x2, y1), (x2-length, y1), color, thick)
    cv2.line(frame, (x2, y1), (x2, y1+length), color, thick)
    # BL
    cv2.line(frame, (x1, y2), (x1+length, y2), color, thick)
    cv2.line(frame, (x1, y2), (x1, y2-length), color, thick)
    # BR
    cv2.line(frame, (x2, y2), (x2-length, y2), color, thick)
    cv2.line(frame, (x2, y2), (x2, y2-length), color, thick)

def draw_glass_panel(frame, x, y, w, h, title, content_lines, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.rectangle(frame, (x, y), (x+w, y+35), color, -1)
    
    # Title text
    text_color = (0, 0, 0) if sum(color) > 350 else (255, 255, 255)
    cv2.putText(frame, title, (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Content text
    cy = y + 60
    for text, scale, tcolor, thick in content_lines:
        texts = text.split('\n')
        for t in texts:
            cv2.putText(frame, t, (x+10, cy), cv2.FONT_HERSHEY_SIMPLEX, scale, tcolor, thick)
            cy += int(scale * 35) + 5
        cy += 5

# ─────────────────────────────────────────────
#  REALTIME WEBCAM MODE
# ─────────────────────────────────────────────
def realtime_predict():
    import time
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[REALTIME] Tekan SPACE untuk predict, Q untuk keluar")
    last_results = None
    
    scan_y_offset = 0
    scan_dir = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame intentionally or process normally
        # frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        box_size = 320
        
        # Base color for scanning
        base_color = (0, 255, 128)
        
        # Animasi scanline
        scan_y_offset += 6 * scan_dir
        if scan_y_offset > box_size:
            scan_y_offset = box_size
            scan_dir = -1
        elif scan_y_offset < 0:
            scan_y_offset = 0
            scan_dir = 1
            
        scan_y = cy - box_size//2 + scan_y_offset

        if last_results:
            top = last_results[0]
            info = top.get("info", {})
            base_color = info.get("color", (255,255,255))
            
        # Draw scanning area bounds
        draw_corners(frame, cx, cy, box_size, base_color, thick=4)
        
        # Glow grid / Overlay inside box
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx-box_size//2, cy-box_size//2), (cx+box_size//2, cy+box_size//2), base_color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            
        # Scanline
        cv2.line(frame, (cx-box_size//2, scan_y), (cx+box_size//2, scan_y), base_color, 2)
        cv2.circle(frame, (cx, scan_y), 4, (255,255,255), -1)

        cv2.putText(frame, "Arahkan daun ke dalam kotak ini",
                    (cx-160, cy-box_size//2-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, base_color, 2)

        # Draw Last Result UI
        if last_results:
            top = last_results[0]
            info = top.get("info", {})
            label = info.get("id", top["class"])
            conf  = top["confidence"] * 100
            severity = info.get("severity", "Aman")
            treatment = info.get("treatment", "Tidak ada")
            
            # Pisahkan treatment jadi beberapa baris jika kepanjangan
            words = treatment.split(' ')
            lines = []
            curr_line = ""
            for word in words:
                if len(curr_line) + len(word) < 45:
                    curr_line += word + " "
                else:
                    lines.append(curr_line)
                    curr_line = word + " "
            if curr_line:
                lines.append(curr_line)
            treatment_fmt = '\n'.join(lines)

            # Panel Utama (Kiri Bawah)
            content_left = [
                (f"Penyakit : {label}", 0.75, base_color, 2),
                (f"Confidence: {conf:.1f}%", 0.65, (255,255,255), 1),
                (f"Keparahan : {severity}", 0.65, (0, 165, 255) if severity != "Tidak ada" else (0,255,0), 1),
                (f"", 0.3, (0,0,0), 1), # spacer
                (f"Penanganan:", 0.6, (200,200,200), 1),
                (treatment_fmt, 0.5, (255,255,255), 1)
            ]
            draw_glass_panel(frame, 20, h-260, 480, 240, "HASIL DETEKSI DETIL", content_left, base_color)
            
            # Panel Kanan (Top 3)
            content_right = []
            for i, r in enumerate(last_results):
                r_info = r.get("info", {})
                r_label = r_info.get("id", r["class"])[:20]
                r_conf = r["confidence"] * 100
                r_color = r_info.get("color", (200,200,200))
                content_right.append((f"{i+1}. {r_label}", 0.6, r_color, 2))
                content_right.append((f"   {r_conf:.1f}%", 0.5, (255,255,255), 1))
                
            draw_glass_panel(frame, w-320, h-260, 300, 240, "TOP 3 PREDIKSI", content_right, (255, 100, 100))

        # Instructions Header Background
        cv2.rectangle(frame, (0, 0), (w, 50), (10, 10, 15), -1)
        cv2.putText(frame, "[] TEKAN SPACE UNTUK ANALISIS []  |  [Q] KELUAR",
                    (w//2 - 250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        cv2.imshow("Advanced Plant Disease Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Crop area tengah dan predict
            roi = frame[cy-box_size//2:cy+box_size//2,
                        cx-box_size//2:cx+box_size//2]
            tmp_path = "temp_predict.jpg"
            cv2.imwrite(tmp_path, roi)
            print("[PREDICT] Analyzing...")
            last_results = predict_image(tmp_path)
            top = last_results[0]
            info = top.get("info", {})
            print(f"[RESULT] {info.get('id', top['class'])} — {top['confidence']*100:.1f}%")

    cap.release()
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  PLANT DISEASE PREDICTOR")
    print("="*50)
    print("  1. Predict dari file gambar")
    print("  2. Realtime webcam")
    print("="*50)

    choice = input("\nPilih mode (1/2): ").strip()

    if choice == "1":
        img_path = input("Path gambar daun: ").strip().strip('"')
        if os.path.exists(img_path):
            print(f"\n[PREDICT] Analyzing {os.path.basename(img_path)}...")
            results = predict_image(img_path)
            print("\n[TOP-3 PREDIKSI]")
            for i, r in enumerate(results):
                info = r.get("info", {})
                print(f"  {i+1}. {info.get('id', r['class']):<30} {r['confidence']*100:.2f}%")
            show_prediction(img_path, results)
        else:
            print(f"[ERROR] File tidak ditemukan: {img_path}")

    elif choice == "2":
        realtime_predict()