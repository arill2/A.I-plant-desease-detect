"""
Definisi model: MobileNetV2 (ImageNet) + classification head, dua tahap latih.

Arsitektur inti sengaja dipertahankan sama dengan versi sebelumnya (512 -> 256,
dropout 0.4 / 0.3) supaya perbandingan hasil masih masuk akal. Yang berubah
adalah cara memanggilnya dan cara memutuskan layer mana yang dibekukan.
"""

from __future__ import annotations


def build_augmentation():
    """Augmentasi ringan yang aman untuk klasifikasi daun.

    Dulu memakai ImageDataGenerator (sudah deprecated di Keras 3). Sekarang
    memakai layer augmentasi yang jalan di dalam graf tf.data, jadi bisa
    diparalelkan dan hasilnya bisa direproduksi lewat seed global.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            # 20 derajat ~ 0.055 putaran penuh
            tf.keras.layers.RandomRotation(0.055, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.2, fill_mode="nearest"),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 255.0)),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_model(num_classes: int, img_size: tuple[int, int], learning_rate: float):
    """Bangun model dan kembalikan (model, base_model)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3), name="leaf_image")

    # training=False penting: BatchNormalization di dalam base tetap memakai
    # statistik ImageNet selama fine-tuning, bukan statistik batch dataset kita.
    # Tanpa ini, akurasi fine-tune sering justru turun.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(512, activation="relu", name="head_fc1")(x)
    x = layers.Dropout(0.4, name="head_drop1")(x)
    x = layers.Dense(256, activation="relu", name="head_fc2")(x)
    x = layers.Dropout(0.3, name="head_drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="plant_disease_mobilenetv2")
    compile_model(model, learning_rate)
    return model, base_model


def compile_model(model, learning_rate: float) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )


def unfreeze_for_finetune(base_model, n_layers: int) -> int:
    """Buka `n_layers` layer terakhir base model, kembalikan jumlah layer trainable."""
    if n_layers <= 0:
        raise ValueError("n_layers harus lebih besar dari 0")
    if n_layers > len(base_model.layers):
        raise ValueError(
            f"n_layers={n_layers} melebihi jumlah layer base model ({len(base_model.layers)})"
        )
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    return sum(1 for layer in base_model.layers if layer.trainable)


def count_trainable_parameters(model) -> int:
    return int(sum(int(v.numpy().size) for v in model.trainable_variables))
