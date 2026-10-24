"""
Training dua tahap + evaluasi pada test set terpisah.

Perbedaan penting dibanding versi lama:

1. Ada tiga split: train / validasi / test. Dulu angka "95%" dihitung pada set
   yang sama yang dipakai ModelCheckpoint dan EarlyStopping, jadi itu nilai
   maksimum yang sudah dipilih, bukan estimasi yang jujur.
2. Gambar duplikat dibuang sebelum split, supaya salinan identik tidak bocor
   ke train dan test sekaligus.
3. Jumlah kelas dibaca dari isi folder dataset, bukan konstanta hardcoded, dan
   setiap kelas wajib punya profil di config.DISEASE_INFO (kalau tidak, gagal
   di awal, bukan diam-diam menghasilkan "Penanganan: Tidak ada").
4. Preprocessing memakai mobilenet_v2.preprocess_input (skala [-1, 1]) yang
   memang diharapkan bobot ImageNet, bukan /255 yang bikin distribusi input
   meleset dari bobot pretrained.

Pakai:
    python train.py --dry-run      # cek split tanpa training (cepat, tanpa GPU)
    python train.py                # training penuh
    python train.py --epochs-head 5 --epochs-finetune 5   # eksperimen cepat
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from data import (
    collect_samples,
    deduplicate,
    discover_classes,
    make_dataset,
    stratified_split,
    summarize,
    validate_classes,
    verify_no_overlap,
)


# ----------------------------------------------------------------- argumen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training Plant Disease Detection")
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--epochs-head", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--unfreeze-layers", type=int, default=config.UNFREEZE_LAYERS)
    parser.add_argument(
        "--preprocess",
        choices=["mobilenet_v2", "rescale"],
        default=config.PREPROCESS,
        help="Preprocessing input. Pakai 'rescale' hanya untuk memuat model lama.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Jangan buang gambar duplikat (hanya untuk perbandingan).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya siapkan dan laporkan split, tanpa memuat TensorFlow.",
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------- persiapan


def prepare_dataset(args: argparse.Namespace):
    """Kembalikan (class_names, split, info) tanpa menyentuh TensorFlow."""
    print(f"[DATA] Dataset: {args.dataset}")

    class_names = discover_classes(args.dataset)
    unknown, missing = validate_classes(class_names, config.DISEASE_INFO)
    if unknown:
        raise SystemExit(
            "[FATAL] Kelas berikut ada di dataset tapi belum punya profil penanganan:\n"
            + "\n".join(f"  - {name}" for name in unknown)
            + "\nTambahkan entri untuk kelas tersebut di config.DISEASE_INFO."
        )
    if missing:
        print(
            "[WARN] Kelas ini didefinisikan di config tapi tidak ada di dataset:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    print(f"[DATA] Jumlah kelas: {len(class_names)}")

    samples = collect_samples(args.dataset, class_names)
    print(f"[DATA] Total gambar terbaca: {len(samples)}")

    removed = 0
    if not args.no_dedupe:
        samples, removed = deduplicate(samples)
        print(f"[DATA] Duplikat isi dibuang: {removed}")
    else:
        print("[DATA] Deduplikasi dimatikan (--no-dedupe)")

    split = stratified_split(
        samples,
        val_fraction=config.VAL_FRACTION,
        test_fraction=config.TEST_FRACTION,
        seed=args.seed,
        label_order=class_names,
    )
    verify_no_overlap(split)

    summary = summarize(split, class_names)
    print("\n[DATA] Sebaran per kelas (train / val / test):")
    for name in class_names:
        row = summary[name]
        print(f"  {name:<48} {row['train']:>5} {row['val']:>5} {row['test']:>5}")

    print(
        f"\n[DATA] Total: train={len(split.train)} "
        f"val={len(split.val)} test={len(split.test)}"
    )

    info = {
        "dataset_dir": str(args.dataset),
        "num_classes": len(class_names),
        "total_images": len(samples) + removed,
        "duplicates_removed": removed,
        "split_sizes": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        "per_class": summary,
        "seed": args.seed,
    }
    return class_names, split, info


# ----------------------------------------------------------------- evaluasi


def evaluate_on_test(model, test_dataset, class_names, report_dir: Path):
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    report_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])

    display = [config.profile_for(name).name_id for name in class_names]
    labels = list(range(len(class_names)))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=display,
        zero_division=0,
        digits=3,
    )
    (report_dir / "test_classification_report.txt").write_text(report_text, encoding="utf-8")
    print("\n[EVAL] Classification report (test set):")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display,
        yticklabels=display,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix - test set", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(display, rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(report_dir / "test_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-kelas F1, dipakai untuk menyorot kelas terlemah di ringkasan.
    per_class = {}
    for index, name in enumerate(class_names):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[index, :].sum()),
        }

    return per_class


def plot_history(all_history: dict[str, list[float]], report_dir: Path, split_epoch: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History - MobileNetV2", fontsize=14, fontweight="bold")

    axes[0].plot(all_history["accuracy"], label="Train", color="#2ecc71", linewidth=2)
    axes[0].plot(all_history["val_accuracy"], label="Validation", color="#e74c3c", linewidth=2)
    axes[0].axvline(x=split_epoch, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(all_history["loss"], label="Train", color="#2ecc71", linewidth=2)
    axes[1].plot(all_history["val_loss"], label="Validation", color="#e74c3c", linewidth=2)
    axes[1].axvline(x=split_epoch, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(report_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- utama


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    class_names, split, dataset_info = prepare_dataset(args)

    if args.dry_run:
        print("\n[DRY-RUN] Split valid. Tidak ada training yang dijalankan.")
        return 0

    import tensorflow as tf

    from model import (
        build_augmentation,
        build_model,
        compile_model,
        count_trainable_parameters,
        unfreeze_for_finetune,
    )

    tf.keras.utils.set_random_seed(args.seed)

    print(f"\n[TRAIN] TensorFlow {tf.__version__}")
    print(f"[TRAIN] GPU: {tf.config.list_physical_devices('GPU')}")

    train_dataset = make_dataset(
        split.train,
        class_names,
        img_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        preprocess=args.preprocess,
        training=True,
        augment=build_augmentation(),
        shuffle=True,
        seed=args.seed,
    )
    val_dataset = make_dataset(
        split.val,
        class_names,
        img_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        preprocess=args.preprocess,
    )
    test_dataset = make_dataset(
        split.test,
        class_names,
        img_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        preprocess=args.preprocess,
    )

    model, base_model = build_model(len(class_names), config.IMG_SIZE, config.LR_HEAD)
    print(f"[MODEL] Parameter trainable: {count_trainable_parameters(model):,}")

    model_dir = config.MODEL_DIR
    report_dir = config.REPORT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    callbacks_head = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_head.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(report_dir / "history_head.csv")),
    ]

    print("\n" + "=" * 60)
    print(f"  TAHAP 1 - classification head ({args.epochs_head} epoch, base frozen)")
    print("=" * 60)
    history_head = model.fit(
        train_dataset,
        epochs=args.epochs_head,
        validation_data=val_dataset,
        callbacks=callbacks_head,
    )

    trainable_base = unfreeze_for_finetune(base_model, args.unfreeze_layers)
    compile_model(model, config.LR_FINETUNE)
    print(
        f"\n[FINETUNE] {trainable_base} layer base dibuka, "
        f"{count_trainable_parameters(model):,} parameter trainable"
    )

    callbacks_finetune = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_finetuned.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(report_dir / "history_finetune.csv")),
    ]

    print("\n" + "=" * 60)
    print(f"  TAHAP 2 - fine-tune ({args.epochs_finetune} epoch)")
    print("=" * 60)
    history_finetune = model.fit(
        train_dataset,
        epochs=args.epochs_finetune,
        validation_data=val_dataset,
        callbacks=callbacks_finetune,
    )

    combined: dict[str, list[float]] = {
        key: history_head.history[key] + history_finetune.history[key]
        for key in history_head.history
        if key in history_finetune.history
    }
    plot_history(combined, report_dir, split_epoch=len(history_head.history["loss"]))

    print("\n" + "=" * 60)
    print("  EVALUASI")
    print("=" * 60)

    val_metrics = model.evaluate(val_dataset, verbose=0, return_dict=True)
    test_metrics = model.evaluate(test_dataset, verbose=0, return_dict=True)

    print(
        f"[VAL ] dipakai untuk memilih checkpoint, jadi angkanya optimistis: "
        f"acc={val_metrics['accuracy'] * 100:.2f}% top3={val_metrics['top3_acc'] * 100:.2f}%"
    )
    print(
        f"[TEST] tidak dipakai untuk keputusan apa pun, ini angka jujurnya:  "
        f"acc={test_metrics['accuracy'] * 100:.2f}% top3={test_metrics['top3_acc'] * 100:.2f}%"
    )

    per_class = evaluate_on_test(model, test_dataset, class_names, report_dir)
    weakest = sorted(per_class.items(), key=lambda item: item[1]["f1"])[:3]
    print("\n[EVAL] Tiga kelas terlemah di test set:")
    for name, metrics in weakest:
        print(f"  {name:<48} F1={metrics['f1']:.3f} support={metrics['support']}")

    model.save(config.MODEL_PATH)
    print(f"\n[SAVED] Model -> {config.MODEL_PATH}")

    metadata = {
        "format_version": config.FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_names": class_names,
        "img_size": list(config.IMG_SIZE),
        "preprocess": args.preprocess,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "dataset": dataset_info,
        "training": {
            "epochs_head": args.epochs_head,
            "epochs_finetune": args.epochs_finetune,
            "batch_size": args.batch_size,
            "learning_rate_head": config.LR_HEAD,
            "learning_rate_finetune": config.LR_FINETUNE,
            "unfreeze_layers": args.unfreeze_layers,
        },
        "metrics": {
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_top3_accuracy": float(val_metrics["top3_acc"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_top3_accuracy": float(test_metrics["top3_acc"]),
            "test_per_class": per_class,
        },
        "environment": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
        },
    }
    config.META_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] Metadata -> {config.META_PATH}")
    print(f"[SAVED] Laporan  -> {report_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
