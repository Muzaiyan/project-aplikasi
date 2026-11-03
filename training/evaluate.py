"""
training/evaluate.py
====================
Evaluasi model pada split val/test: mAP50, mAP50-95, precision, recall, dan
metrik per kelas. Hasil juga disimpan ke results/evaluasi.json.

Contoh:
    python training/evaluate.py
    python training/evaluate.py --split val --weights models/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "training" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluasi model deteksi jalan rusak")
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"[ERROR] Bobot tidak ditemukan: {weights}\nLatih model dulu: python training/train.py")

    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    cfg["path"] = str((ROOT / "dataset").resolve())
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data_abs = RUNS_DIR / "_data_abs.yaml"
    data_abs.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(
        data=str(data_abs), split=args.split, imgsz=args.imgsz, conf=args.conf,
        device=args.device, project=str(RUNS_DIR), name=f"eval_{args.split}", exist_ok=True, plots=True,
    )

    names = model.names
    per_class = {}
    for i, cls_idx in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(cls_idx)]] = {
            "precision": round(float(p), 4), "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4),
        }

    report = {
        "split": args.split,
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "per_class": per_class,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "evaluasi.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== HASIL EVALUASI ===")
    print(f"Split      : {args.split}")
    print(f"Precision  : {report['precision']:.3f}")
    print(f"Recall     : {report['recall']:.3f}")
    print(f"mAP@50     : {report['mAP50']:.3f}")
    print(f"mAP@50-95  : {report['mAP50-95']:.3f}")
    for k, v in per_class.items():
        print(f"  - {k:<20} P={v['precision']:.3f} R={v['recall']:.3f} mAP50={v['mAP50']:.3f}")
    print(f"\nLaporan JSON: {out}")


if __name__ == "__main__":
    main()
