import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import threading
import urllib.request
import math
import numpy as np
from collections import deque
from gtts import gTTS
from playsound import playsound

# ─────────────────────────────────────────────
#  DOWNLOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("[MODEL] Downloading hand_landmarker.task ...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[MODEL] Download selesai!\n")

# ─────────────────────────────────────────────
#  KONFIGURASI SIGNS
# ─────────────────────────────────────────────
SIGNS = [
    {"label": "Sign 1", "word": "Perkenalkan", "color": (0, 180, 255),   "emoji": "🖐️", "icon": "OPEN"},
    {"label": "Sign 2", "word": "Nama saya",   "color": (0, 255, 128),   "emoji": "☝️", "icon": "INDEX"},
    {"label": "Sign 3", "word": "Muhammad",     "color": (255, 100, 255), "emoji": "✌️", "icon": "PEACE"},
    {"label": "Sign 4", "word": "Syahrir",      "color": (255, 200, 0),   "emoji": "🤘", "icon": "ROCK"},
    {"label": "Sign 5", "word": "Hamdani",      "color": (100, 100, 255), "emoji": "✊", "icon": "FIST"},
]

TTS_DIR = "tts_cache"
os.makedirs(TTS_DIR, exist_ok=True)

def generate_audio():
    for sign in SIGNS:
        path = os.path.join(TTS_DIR, f"{sign['word'].replace(' ', '_')}.mp3")
        if not os.path.exists(path):
            print(f"[TTS] Generating: {sign['word']} ...")
            tts = gTTS(text=sign["word"], lang="id")
            tts.save(path)
    print("[TTS] Semua audio siap!\n")

generate_audio()

def play_audio(word):
    def _play():
        path = os.path.join(TTS_DIR, f"{word.replace(' ', '_')}.mp3")
        if os.path.exists(path):
            playsound(path)
    threading.Thread(target=_play, daemon=True).start()

# ─────────────────────────────────────────────
#  VISUAL EFFECTS - PARTICLE SYSTEM
# ─────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, lifetime=1.5):
        self.x = x
        self.y = y
        angle = np.random.uniform(0, 2 * math.pi)
        speed = np.random.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 3
        self.color = color
        self.lifetime = lifetime
        self.born = time.time()
        self.size = np.random.randint(3, 8)

    def update(self):
        self.x += self.vx
        self.vy += 0.15  # gravity
        self.y += self.vy
        self.vx *= 0.98

    def draw(self, frame):
        age = time.time() - self.born
        alpha = max(0, 1 - age / self.lifetime)
        if alpha <= 0:
            return False
        sz = max(1, int(self.size * alpha))
        c = tuple(max(0, min(255, int(ch * alpha))) for ch in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), sz, c, -1)
        # glow effect
        cv2.circle(frame, (int(self.x), int(self.y)), sz + 2,
                   tuple(max(0, min(255, int(ch * alpha * 0.3))) for ch in self.color), 1)
        return True

    def alive(self):
        return (time.time() - self.born) < self.lifetime


particles = []

def spawn_particles(x, y, color, count=30):
    for _ in range(count):
        particles.append(Particle(x, y, color))

def update_particles(frame):
    global particles
    alive = []
    for p in particles:
        p.update()
        if p.draw(frame):
            alive.append(p)
    particles = alive

# ─────────────────────────────────────────────
#  TRAIL EFFECT - Hand movement trail
# ─────────────────────────────────────────────
hand_trail = deque(maxlen=40)

def draw_trail(frame, trail, color):
    """Draw a glowing trail following hand movement."""
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        thickness = max(1, int(alpha * 5))
        c = tuple(max(0, min(255, int(ch * alpha))) for ch in color)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness)

# ─────────────────────────────────────────────
#  NEON GLOW TEXT
# ─────────────────────────────────────────────
def draw_neon_text(frame, text, pos, font, scale, color, thickness):
    """Draw text with neon glow effect."""
    x, y = pos
    # Outer glow layers
    for i in range(5, 0, -1):
        glow_color = tuple(min(255, int(ch * 0.3)) for ch in color)
        cv2.putText(frame, text, (x, y), font, scale, glow_color, thickness + i * 2)
    # Main text
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    # Bright inner
    bright = tuple(min(255, ch + 80) for ch in color)
    cv2.putText(frame, text, (x, y), font, scale, bright, max(1, thickness - 1))

# ─────────────────────────────────────────────
#  ANIMATED PROGRESS RING
# ─────────────────────────────────────────────
def draw_progress_ring(frame, center, radius, progress, color, thickness=4):
    """Draw a circular progress indicator."""
    # Background ring
    cv2.circle(frame, center, radius, (40, 40, 40), thickness)
    # Progress arc
    angle = int(progress * 360)
    if angle > 0:
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, color, thickness + 2)
        # Tip glow
        tip_angle = math.radians(-90 + angle)
        tip_x = int(center[0] + radius * math.cos(tip_angle))
        tip_y = int(center[1] + radius * math.sin(tip_angle))
        cv2.circle(frame, (tip_x, tip_y), thickness + 3, color, -1)
        cv2.circle(frame, (tip_x, tip_y), thickness + 6,
                   tuple(min(255, int(ch * 0.5)) for ch in color), 2)

# ─────────────────────────────────────────────
#  GLASSMORPHISM PANEL
# ─────────────────────────────────────────────
def draw_glass_panel(frame, x1, y1, x2, y2, alpha=0.3, border_color=(255, 255, 255)):
    """Draw a frosted glass panel effect."""
    overlay = frame.copy()
    # Blurred background for frosted effect
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        blurred = cv2.GaussianBlur(roi, (21, 21), 0)
        tinted = cv2.addWeighted(blurred, 0.7, np.full_like(roi, (30, 30, 40)), 0.3, 0)
        overlay[y1:y2, x1:x2] = tinted
    frame[:] = cv2.addWeighted(overlay, alpha + 0.5, frame, 0.5 - alpha, 0)
    # Border with rounded feel
    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  tuple(int(ch * 0.4) for ch in border_color), 1, cv2.LINE_AA)
    # Top highlight line
    cv2.line(frame, (x1 + 2, y1 + 1), (x2 - 2, y1 + 1),
             tuple(min(255, int(ch * 0.6)) for ch in border_color), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  PULSING CIRCLE INDICATOR
# ─────────────────────────────────────────────
def draw_pulse(frame, center, base_radius, color, phase):
    """Draw a pulsing circle effect."""
    for i in range(3):
        r = max(1, base_radius + int(15 * math.sin(phase + i * 0.5)))
        alpha = max(0, 0.5 - i * 0.15)
        c = tuple(max(0, min(255, int(ch * alpha))) for ch in color)
        cv2.circle(frame, center, r, c, 2, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  DRAW HAND LANDMARKS (ENHANCED)
# ─────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Finger groups for rainbow coloring
FINGER_GROUPS = {
    "thumb":  [(0,1),(1,2),(2,3),(3,4)],
    "index":  [(0,5),(5,6),(6,7),(7,8)],
    "middle": [(5,9),(9,10),(10,11),(11,12)],
    "ring":   [(9,13),(13,14),(14,15),(15,16)],
    "pinky":  [(13,17),(17,18),(18,19),(19,20)],
    "palm":   [(0,17)],
}

FINGER_COLORS = {
    "thumb":  (0, 200, 255),    # orange
    "index":  (0, 255, 128),    # green
    "middle": (255, 255, 0),    # cyan
    "ring":   (255, 100, 255),  # magenta
    "pinky":  (255, 100, 100),  # light blue
    "palm":   (200, 200, 200),  # gray
}

def draw_landmarks_enhanced(frame, landmarks, w, h, detected_color=None, phase=0):
    """Draw enhanced hand landmarks with per-finger colors and glow."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw connections per finger group with glow
    for group_name, connections in FINGER_GROUPS.items():
        color = detected_color if detected_color else FINGER_COLORS[group_name]
        for a, b in connections:
            # Glow line
            cv2.line(frame, pts[a], pts[b],
                     tuple(int(c * 0.3) for c in color), 6, cv2.LINE_AA)
            # Main line
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    # Draw landmark points with pulse
    TIPS = [4, 8, 12, 16, 20]
    for i, pt in enumerate(pts):
        if i in TIPS:
            # Fingertip: pulsing glow
            pulse_r = max(4, 7 + int(3 * math.sin(phase * 3 + i)))
            cv2.circle(frame, pt, pulse_r + 4,
                       (0, 51, 51), -1)
            cv2.circle(frame, pt, pulse_r, (0, 255, 255), -1)
            cv2.circle(frame, pt, max(1, pulse_r - 2), (255, 255, 255), -1)
        elif i == 0:
            # Wrist: bigger
            cv2.circle(frame, pt, 8, (100, 255, 100), -1)
            cv2.circle(frame, pt, 10, (100, 255, 100), 2, cv2.LINE_AA)
        else:
            # Joint
            cv2.circle(frame, pt, 4, (180, 180, 255), -1)

    return pts

# ─────────────────────────────────────────────
#  MINI HAND ICON DRAWING
# ─────────────────────────────────────────────
def draw_mini_hand(frame, cx, cy, fingers_state, size=20, color=(200, 200, 200)):
    """Draw a small iconic hand showing finger states."""
    # Palm
    cv2.circle(frame, (cx, cy + size // 3), size // 2, color, 1, cv2.LINE_AA)

    # Finger positions (relative to palm center)
    finger_angles = [-70, -50, -30, -10, 10]  # degrees
    for i, (angle_deg, up) in enumerate(zip([-60, -35, -15, 5, 25], fingers_state)):
        angle = math.radians(angle_deg - 90)
        length = size if up else size // 3
        ex = int(cx + length * math.cos(angle))
        ey = int(cy + size // 3 + length * math.sin(angle))
        c = (0, 255, 128) if up else (80, 80, 80)
        cv2.line(frame, (cx, cy + size // 3), (ex, ey), c, 2, cv2.LINE_AA)
        if up:
            cv2.circle(frame, (ex, ey), 3, c, -1)

# ─────────────────────────────────────────────
#  DETEKSI JARI
# ─────────────────────────────────────────────
def fingers_up(lm, handedness="Right"):
    result = []
    if handedness == "Right":
        result.append(1 if lm[4].x < lm[2].x else 0)
    else:
        result.append(1 if lm[4].x > lm[2].x else 0)

    finger_tips = [8, 12, 16, 20]
    finger_mcp  = [5,  9, 13, 17]
    for tip, mcp in zip(finger_tips, finger_mcp):
        result.append(1 if lm[tip].y < lm[mcp].y else 0)
    return result

# ─────────────────────────────────────────────
#  SIGN PATTERNS
# ─────────────────────────────────────────────
SIGN_PATTERNS = [
    [1, 1, 1, 1, 1],  # Perkenalkan - open palm
    [0, 1, 0, 0, 0],  # Nama saya   - index only
    [0, 1, 1, 0, 0],  # Muhammad    - peace
    [1, 1, 0, 0, 1],  # Syahrir     - rock
    [0, 0, 0, 0, 0],  # Hamdani     - fist
]

SIGN_NAMES = ["Perkenalkan", "Nama saya", "Muhammad", "Syahrir", "Hamdani"]

def detect_sign(fingers):
    for i, pattern in enumerate(SIGN_PATTERNS):
        if fingers == pattern:
            return i
    return -1

# ─────────────────────────────────────────────
#  SENTENCE BUILDER & HISTORY
# ─────────────────────────────────────────────
sentence_words = []
sentence_display_time = 0
recognition_history = deque(maxlen=10)

def add_to_sentence(word, color):
    sentence_words.append({"word": word, "color": color, "time": time.time()})

def get_sentence_text():
    return " ".join(w["word"] for w in sentence_words)

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
last_sign_idx    = -1
sign_hold_start  = None
HOLD_DURATION    = 1.2
last_spoken_time = 0
COOLDOWN         = 2.5
display_text     = ""
display_color    = (255, 255, 255)
text_show_until  = 0
frame_count      = 0
fps_time         = time.time()
fps_val          = 0

# Visual modes
VISUAL_MODES = ["NEON", "MATRIX", "FROST"]
current_mode = 0

# Matrix rain effect
matrix_columns = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("=" * 55)
print("  ✨ HAND SIGN PERKENALAN — Muh Syahrir Hamdani ✨")
print("  Enhanced Edition v2.0")
print("=" * 55)
print("  Q       = Keluar")
print("  M       = Ganti mode visual (NEON / MATRIX / FROST)")
print("  C       = Hapus kalimat")
print("  S       = Screenshot")
print("  SPACE   = Speak kalimat lengkap")
print("=" * 55)
print("  Sign 1  🖐️  Open palm     -> Perkenalkan")
print("  Sign 2  ☝️  Telunjuk saja -> Nama saya")
print("  Sign 3  ✌️  Peace sign    -> Muhammad")
print("  Sign 4  🤘 Rock/Love     -> Syahrir")
print("  Sign 5  ✊ Kepalan       -> Hamdani\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    now   = time.time()
    phase = now * 2  # animation phase

    # Init matrix columns
    if not matrix_columns:
        matrix_columns = [{"y": np.random.randint(-h, 0), "speed": np.random.randint(5, 15),
                           "x": i * 14} for i in range(w // 14)]

    # ── BACKGROUND EFFECTS ──
    mode_name = VISUAL_MODES[current_mode]

    if mode_name == "MATRIX":
        # Matrix rain overlay
        overlay_matrix = np.zeros_like(frame)
        for col in matrix_columns:
            char = chr(np.random.randint(0x30A0, 0x30FF))
            intensity = np.random.randint(100, 255)
            cv2.putText(overlay_matrix, char, (col["x"], int(col["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, intensity, 0), 1)
            col["y"] += col["speed"]
            if col["y"] > h:
                col["y"] = np.random.randint(-50, 0)
                col["speed"] = np.random.randint(5, 15)
        frame = cv2.add(frame, overlay_matrix)

    elif mode_name == "FROST":
        # Cool blue tint
        blue_tint = np.full_like(frame, (50, 20, 5))
        frame = cv2.addWeighted(frame, 0.85, blue_tint, 0.15, 0)

    # ── VIGNETTE EFFECT ──
    rows, cols = frame.shape[:2]
    X = cv2.getGaussianKernel(cols, cols * 0.6)
    Y = cv2.getGaussianKernel(rows, rows * 0.6)
    M = Y * X.T
    M = M / M.max()
    for c_ch in range(3):
        frame[:, :, c_ch] = np.uint8(frame[:, :, c_ch] * M)

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = detector.detect(mp_image)

    detected_idx = -1
    fingers      = [0, 0, 0, 0, 0]

    if detection.hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(detection.hand_landmarks):
            hand_label = "Right"
            if detection.handedness and hand_idx < len(detection.handedness):
                hand_label = detection.handedness[hand_idx][0].category_name

            det_color = None
            f = fingers_up(hand_landmarks, hand_label)
            idx = detect_sign(f)
            if idx != -1:
                det_color = SIGNS[idx]["color"]

            pts = draw_landmarks_enhanced(frame, hand_landmarks, w, h, det_color, phase)

            if hand_idx == 0:
                fingers = f
                detected_idx = idx

                # Add wrist to trail
                wrist = pts[0]
                hand_trail.append(wrist)

                # Add mid-hand point to trail
                mid_x = (pts[0][0] + pts[9][0]) // 2
                mid_y = (pts[0][1] + pts[9][1]) // 2

    # Draw hand trail
    trail_color = SIGNS[detected_idx]["color"] if detected_idx != -1 else (100, 200, 255)
    draw_trail(frame, list(hand_trail), trail_color)

    # Update particles
    update_particles(frame)

    # ── TOP BAR with title ──
    draw_glass_panel(frame, 0, 0, w, 45, alpha=0.3, border_color=(100, 200, 255))
    title_text = f"HAND SIGN PERKENALAN  |  Mode: {mode_name}  |  FPS: {fps_val}"
    cv2.putText(frame, title_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1, cv2.LINE_AA)

    # ── DEBUG PANEL — kiri atas ──
    panel_h = 230
    draw_glass_panel(frame, 10, 55, 310, 55 + panel_h, alpha=0.25,
                     border_color=(0, 255, 200))

    cv2.putText(frame, "FINGER STATUS", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (20, 87), (170, 87), (0, 255, 200), 1)

    finger_labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    finger_icons  = ["[T]", "[I]", "[M]", "[R]", "[P]"]
    for i, (label, val) in enumerate(zip(finger_labels, fingers)):
        y_pos = 110 + i * 28
        # Status indicator box
        box_color = (0, 255, 100) if val == 1 else (60, 60, 80)
        cv2.rectangle(frame, (20, y_pos - 14), (38, y_pos + 4), box_color, -1)
        cv2.rectangle(frame, (20, y_pos - 14), (38, y_pos + 4), (200, 200, 200), 1)

        text_color = (0, 255, 100) if val == 1 else (100, 100, 100)
        status = "UP" if val else "--"
        cv2.putText(frame, f"{finger_icons[i]} {label}: {status}",
                    (45, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    pattern_str = str(fingers)
    cv2.putText(frame, f"Pattern: {pattern_str}", (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)

    # ── SIGN GUIDE — kanan atas ──
    guide_w = 290
    guide_x = w - guide_w - 10
    guide_h = 230
    draw_glass_panel(frame, guide_x, 55, w - 10, 55 + guide_h, alpha=0.25,
                     border_color=(255, 200, 0))

    cv2.putText(frame, "SIGN GUIDE", (guide_x + 10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (guide_x + 10, 87), (guide_x + 130, 87), (255, 200, 0), 1)

    guide_data = [
        ("Open Palm", [1,1,1,1,1]),
        ("Telunjuk",  [0,1,0,0,0]),
        ("Peace",     [0,1,1,0,0]),
        ("Rock",      [1,1,0,0,1]),
        ("Kepalan",   [0,0,0,0,0]),
    ]
    for i, (gname, gpattern) in enumerate(guide_data):
        y_pos = 110 + i * 28
        is_active = detected_idx == i
        col = SIGNS[i]["color"] if is_active else (120, 120, 120)

        # Active indicator
        if is_active:
            cv2.circle(frame, (guide_x + 15, y_pos - 5), 5, SIGNS[i]["color"], -1)
            cv2.circle(frame, (guide_x + 15, y_pos - 5), 8, SIGNS[i]["color"], 1)
        else:
            cv2.circle(frame, (guide_x + 15, y_pos - 5), 4, (80, 80, 80), 1)

        # Mini hand icon
        draw_mini_hand(frame, guide_x + 45, y_pos - 8, gpattern, size=10, color=col)

        cv2.putText(frame, f"{SIGNS[i]['word']}", (guide_x + 65, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Pattern on the right
        cv2.putText(frame, f"{gpattern}", (guide_x + 175, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    # ── LOGIKA HOLD ──
    if detected_idx != -1:
        if detected_idx == last_sign_idx:
            if sign_hold_start and (now - sign_hold_start >= HOLD_DURATION):
                if now - last_spoken_time >= COOLDOWN:
                    sign             = SIGNS[detected_idx]
                    display_text     = sign["word"]
                    display_color    = sign["color"]
                    text_show_until  = now + 3.0
                    last_spoken_time = now
                    sign_hold_start  = None

                    # Add to sentence
                    add_to_sentence(sign["word"], sign["color"])
                    recognition_history.append((sign["word"], now))

                    # Spawn celebration particles
                    spawn_particles(w // 2, h // 2, sign["color"], 50)

                    play_audio(sign["word"])
                    print(f"[SIGN] {sign['label']} -> {sign['word']}")
                    print(f"[KALIMAT] {get_sentence_text()}")
        else:
            last_sign_idx   = detected_idx
            sign_hold_start = now
    else:
        last_sign_idx   = -1
        sign_hold_start = None

    # ── CIRCULAR PROGRESS RING ──
    if sign_hold_start and detected_idx != -1:
        progress = min((now - sign_hold_start) / HOLD_DURATION, 1.0)
        ring_center = (w // 2, h - 90)
        draw_progress_ring(frame, ring_center, 35, progress,
                           SIGNS[detected_idx]["color"], thickness=4)

        # Progress percentage text
        pct = int(progress * 100)
        (tw, _), _ = cv2.getTextSize(f"{pct}%", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, f"{pct}%", (ring_center[0] - tw // 2, ring_center[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, SIGNS[detected_idx]["color"], 2, cv2.LINE_AA)

        # Also draw linear progress bar at the very bottom
        bar_y = h - 8
        cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, bar_y), (int(w * progress), h),
                      SIGNS[detected_idx]["color"], -1)

    # ── BIG DISPLAY TEXT (center) with animation ──
    if now < text_show_until and display_text:
        age = now - (text_show_until - 3.0)
        # Scale animation: pop in then settle
        if age < 0.3:
            scale_factor = 0.5 + 2.0 * (age / 0.3)
        elif age < 0.5:
            scale_factor = 2.5 - 0.5 * ((age - 0.3) / 0.2)
        else:
            scale_factor = 2.0

        # Fade out in last 0.5s
        fade = min(1.0, (text_show_until - now) / 0.5)

        fs = scale_factor
        tk = 4
        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, fs, tk)
        tx = (w - tw) // 2
        ty = h // 2 + th // 2 - 20

        color_faded = tuple(int(ch * fade) for ch in display_color)

        # Shadow / outline
        draw_neon_text(frame, display_text, (tx, ty),
                       cv2.FONT_HERSHEY_DUPLEX, fs, color_faded, tk)

        # Subtitle: sign icon name
        if detected_idx != -1 or display_text:
            sub = f"[ {display_text} ]"
            (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            sx = (w - sw) // 2
            cv2.putText(frame, sub, (sx, ty + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        tuple(int(ch * fade * 0.7) for ch in display_color), 1, cv2.LINE_AA)

    # ── SENTENCE BAR (bottom area) ──
    if sentence_words:
        sent_panel_y = h - 170
        draw_glass_panel(frame, 10, sent_panel_y, w - 10, sent_panel_y + 45,
                         alpha=0.25, border_color=(200, 200, 255))

        cv2.putText(frame, "KALIMAT:", (20, sent_panel_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)

        x_offset = 110
        for wd in sentence_words:
            cv2.putText(frame, wd["word"], (x_offset, sent_panel_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, wd["color"], 2, cv2.LINE_AA)
            (ww, _), _ = cv2.getTextSize(wd["word"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x_offset += ww + 15

    # ── STATUS BAR (bottom) ──
    status_y = h - 55
    draw_glass_panel(frame, 0, status_y, w, h, alpha=0.3, border_color=(80, 80, 120))

    if detected_idx == -1:
        if not detection.hand_landmarks:
            hand_status = "Belum ada tangan terdeteksi - tunjukkan tangan ke kamera"
            st_color = (100, 100, 200)
        else:
            hand_status = "Sign tidak dikenal - coba posisi lain"
            st_color = (100, 150, 255)

        # Animated dots
        dots = "." * (int(now * 2) % 4)
        cv2.putText(frame, f"{hand_status}{dots}", (20, status_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, st_color, 1, cv2.LINE_AA)

        # Pulsing circle for "waiting"
        draw_pulse(frame, (w - 40, status_y + 25), 10, st_color, phase)
    else:
        sign = SIGNS[detected_idx]
        cv2.putText(frame, f">> {sign['word']}  --  Tahan untuk konfirmasi",
                    (20, status_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sign["color"], 2, cv2.LINE_AA)

        # Active indicator pulse
        draw_pulse(frame, (w - 40, status_y + 25), 10, sign["color"], phase)

    # Controls hint
    cv2.putText(frame, "Q:Keluar  M:Mode  C:Clear  S:Screenshot  SPACE:Speak",
                (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    # ── FPS COUNTER ──
    frame_count += 1
    if now - fps_time >= 1.0:
        fps_val = frame_count
        frame_count = 0
        fps_time = now

    cv2.imshow("Hand Sign Perkenalan v2.0 - Enhanced", frame)

    # ── KEY HANDLING ──
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        current_mode = (current_mode + 1) % len(VISUAL_MODES)
        print(f"[MODE] Switched to: {VISUAL_MODES[current_mode]}")
    elif key == ord('c'):
        sentence_words.clear()
        print("[CLEAR] Kalimat dihapus")
    elif key == ord('s'):
        screenshot_path = f"screenshot_{int(now)}.png"
        cv2.imwrite(screenshot_path, frame)
        print(f"[SCREENSHOT] Saved: {screenshot_path}")
    elif key == ord(' '):
        if sentence_words:
            full_sentence = get_sentence_text()
            print(f"[SPEAK] {full_sentence}")
            def speak_sentence(text):
                tts = gTTS(text=text, lang="id")
                tmp_path = os.path.join(TTS_DIR, "sentence_tmp.mp3")
                tts.save(tmp_path)
                playsound(tmp_path)
            threading.Thread(target=speak_sentence, args=(full_sentence,), daemon=True).start()

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\n[INFO] Program selesai. Terima kasih! ✨")