"""
training/train.py
=================
Melatih model YOLO untuk deteksi kerusakan jalan, lalu menyalin bobot terbaik
ke models/best.pt agar langsung dipakai aplikasi web.

Contoh:
    python training/train.py                         # default: yolov8n, 100 epoch
    python training/train.py --model yolov8s.pt --epochs 150 --batch 16
    python training/train.py --device cpu --epochs 30 --imgsz 512
    python training/train.py --resume                # lanjutkan training terakhir
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "training" / "runs"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.glob("*") if p.suffix.lower() in IMG_EXT) if folder.exists() else 0


def prepare_data_yaml() -> Path:
    """Buat salinan data.yaml dengan path absolut (menghindari masalah working directory)."""
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    cfg["path"] = str((ROOT / "dataset").resolve())
    out = RUNS_DIR / "_data_abs.yaml"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def check_dataset() -> None:
    ds = ROOT / "dataset"
    n_train = count_images(ds / "images" / "train")
    n_val = count_images(ds / "images" / "val")
    print(f"[dataset] train={n_train} gambar | val={n_val} gambar | test={count_images(ds / 'images' / 'test')} gambar")
    if n_train == 0 or n_val == 0:
        sys.exit(
            "\n[ERROR] Dataset kosong. Isi dulu:\n"
            "  dataset/images/{train,val,test}/  -> gambar (.jpg/.png)\n"
            "  dataset/labels/{train,val,test}/  -> label YOLO (.txt) dengan nama sama\n"
            "Format label per baris: <class_id> <x_center> <y_center> <width> <height> (nilai 0-1)\n"
            "Sumber: Roboflow Universe (export format YOLOv8) atau RDD2022."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Latih YOLO untuk deteksi jalan rusak")
    ap.add_argument("--model", default="yolov8n.pt", help="bobot awal / arsitektur (yolov8n.pt, yolov8s.pt, yolo11n.pt, ...)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = otomatis")
    ap.add_argument("--device", default=None, help="'0' untuk GPU pertama, 'cpu' untuk CPU (default: otomatis)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30, help="early stopping")
    ap.add_argument("--name", default="road_damage")
    ap.add_argument("--resume", action="store_true", help="lanjutkan dari last.pt pada run --name")
    args = ap.parse_args()

    check_dataset()
    data_yaml = prepare_data_yaml()

    from ultralytics import YOLO

    if args.resume:
        last = RUNS_DIR / args.name / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"[ERROR] {last} tidak ditemukan, tidak bisa resume.")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=str(RUNS_DIR),
            name=args.name,
            exist_ok=True,
            # augmentasi yang cocok untuk citra jalan
            hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
            degrees=5.0, translate=0.1, scale=0.5,
            fliplr=0.5, flipud=0.0, mosaic=1.0,
            plots=True,
        )

    best = RUNS_DIR / args.name / "weights" / "best.pt"
    if best.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        shutil.copy2(best, MODELS_DIR / "best.pt")
        print(f"\n[OK] Model terbaik disalin ke: {MODELS_DIR / 'best.pt'}")
        print("     Jalankan aplikasi:  python app.py")
    else:
        print("[PERINGATAN] best.pt tidak ditemukan di folder run.")


if __name__ == "__main__":
    main()
