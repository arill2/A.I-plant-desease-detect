import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, TensorBoard)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATASET_DIR  = r"D:\ALL PROJECT A.I\dataset\PlantVillage"
MODEL_DIR    = r"D:\ALL PROJECT A.I\models"
LOG_DIR      = r"D:\ALL PROJECT A.I\logs"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_FROZEN   = 10   # tahap 1: hanya train head
EPOCHS_FINETUNE = 10   # tahap 2: fine-tune layer atas
LEARNING_RATE   = 1e-3
NUM_CLASSES     = 15

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

print("=" * 60)
print("  PLANT DISEASE DETECTION — MobileNetV2 Transfer Learning")
print("  Dataset: PlantVillage (Tomato, 10 kelas)")
print("=" * 60)
print(f"\n[CONFIG] Image size : {IMG_SIZE}")
print(f"[CONFIG] Batch size : {BATCH_SIZE}")
print(f"[CONFIG] GPU        : {tf.config.list_physical_devices('GPU')}\n")

# ─────────────────────────────────────────────
#  DATA AUGMENTATION & GENERATOR
# ─────────────────────────────────────────────
print("[DATA] Menyiapkan data generator...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

CLASS_NAMES = list(train_generator.class_indices.keys())
print(f"\n[DATA] Kelas yang ditemukan ({len(CLASS_NAMES)}):")
for i, name in enumerate(CLASS_NAMES):
    print(f"       {i:2d}. {name}")

print(f"\n[DATA] Training  : {train_generator.samples} gambar")
print(f"[DATA] Validation: {val_generator.samples} gambar")

# ─────────────────────────────────────────────
#  SAVE CLASS NAMES
# ─────────────────────────────────────────────
import json
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)
print(f"\n[SAVED] class_names.json -> {MODEL_DIR}")

# ─────────────────────────────────────────────
#  BUILD MODEL — MobileNetV2 + Custom Head
# ─────────────────────────────────────────────
print("\n[MODEL] Membangun model MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze semua layer base dulu

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

model.summary()
total_params = model.count_params()
print(f"\n[MODEL] Total parameters : {total_params:,}")
print(f"[MODEL] Trainable params : {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")

# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────
callbacks_frozen = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    TensorBoard(log_dir=LOG_DIR)
]

# ─────────────────────────────────────────────
#  TAHAP 1: TRAIN HEAD (base frozen)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  TAHAP 1: Training classification head (base frozen)")
print("="*60)

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_FROZEN,
    validation_data=val_generator,
    callbacks=callbacks_frozen,
    verbose=1
)

# ─────────────────────────────────────────────
#  TAHAP 2: FINE-TUNE (unfreeze top layers)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  TAHAP 2: Fine-tuning (unfreeze top 30 layers)")
print("="*60)

base_model.trainable = True
# Freeze semua kecuali 30 layer terakhir
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

trainable_after = sum([tf.size(v).numpy() for v in model.trainable_variables])
print(f"[FINETUNE] Trainable params sekarang: {trainable_after:,}")

callbacks_finetune = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model_finetuned.keras"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_generator,
    callbacks=callbacks_finetune,
    verbose=1
)

# ─────────────────────────────────────────────
#  EVALUASI FINAL
# ─────────────────────────────────────────────
print("\n[EVAL] Evaluasi model pada validation set...")
loss, acc, top3 = model.evaluate(val_generator, verbose=1)
print(f"\n[RESULT] Val Loss     : {loss:.4f}")
print(f"[RESULT] Val Accuracy : {acc*100:.2f}%")
print(f"[RESULT] Top-3 Acc    : {top3*100:.2f}%")

# ─────────────────────────────────────────────
#  CLASSIFICATION REPORT
# ─────────────────────────────────────────────
print("\n[EVAL] Generating classification report...")
val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

short_names = [n.replace("Tomato_", "").replace("Tomato__", "")[:20] for n in CLASS_NAMES]
print("\n" + classification_report(y_true, y_pred, target_names=short_names))

# ─────────────────────────────────────────────
#  PLOT: Training History
# ─────────────────────────────────────────────
def combine_history(h1, h2, key):
    return h1.history[key] + h2.history[key]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History — Plant Disease MobileNetV2", fontsize=14, fontweight='bold')

# Accuracy
acc_train = combine_history(history1, history2, 'accuracy')
acc_val   = combine_history(history1, history2, 'val_accuracy')
axes[0].plot(acc_train, label='Train Accuracy', color='#2ecc71', linewidth=2)
axes[0].plot(acc_val,   label='Val Accuracy',   color='#e74c3c', linewidth=2)
axes[0].axvline(x=len(history1.history['accuracy'])-1,
                color='gray', linestyle='--', alpha=0.5, label='Fine-tune start')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss
loss_train = combine_history(history1, history2, 'loss')
loss_val   = combine_history(history1, history2, 'val_loss')
axes[1].plot(loss_train, label='Train Loss', color='#2ecc71', linewidth=2)
axes[1].plot(loss_val,   label='Val Loss',   color='#e74c3c', linewidth=2)
axes[1].axvline(x=len(history1.history['loss'])-1,
                color='gray', linestyle='--', alpha=0.5, label='Fine-tune start')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "training_history.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"[SAVED] training_history.png -> {MODEL_DIR}")

# ─────────────────────────────────────────────
#  PLOT: Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names,
            linewidths=0.5)
plt.title('Confusion Matrix — Plant Disease Detection', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"[SAVED] confusion_matrix.png -> {MODEL_DIR}")

# ─────────────────────────────────────────────
#  SAVE FINAL MODEL
# ─────────────────────────────────────────────
final_path = os.path.join(MODEL_DIR, "plant_disease_final.keras")
model.save(final_path)
print(f"\n[SAVED] Final model -> {final_path}")

print("\n" + "="*60)
print(f"  TRAINING SELESAI!")
print(f"  Val Accuracy : {acc*100:.2f}%")
print(f"  Top-3 Acc    : {top3*100:.2f}%")
print(f"  Model saved  : {final_path}")
print("="*60)
