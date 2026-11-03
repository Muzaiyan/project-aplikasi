"""
detector.py
===========
Pembungkus model YOLO (Ultralytics) untuk mendeteksi kerusakan jalan.

Kelas yang dideteksi mengikuti dataset/data.yaml (saat ini 1 kelas: damage).
Bila dataset memakai kelas lain, tambahkan namanya (persis sama dengan di
data.yaml) pada CLASS_INFO di bawah agar label & warnanya tampil rapi.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best.pt"

# ---------------------------------------------------------------------------
# Metadata kelas
# ---------------------------------------------------------------------------
CLASS_INFO = {
    "damage": {"label": "Kerusakan Jalan", "color": "#dc2626"},
    # Contoh bila memakai dataset multi-kelas:
    # "longitudinal_crack": {"label": "Retak Memanjang", "color": "#2563eb"},
    # "transverse_crack":   {"label": "Retak Melintang", "color": "#0d9488"},
    # "alligator_crack":    {"label": "Retak Buaya",     "color": "#d97706"},
    # "pothole":            {"label": "Lubang",          "color": "#dc2626"},
}

SEVERITY_LABELS = ["Baik", "Ringan", "Sedang", "Berat"]

# Batas keyakinan tetap (mode sensitif): temuan dengan skor >= 15% ditampilkan.
# Kisaran yang disarankan 10-20%. Untuk mengubah tanpa menyunting kode:
#   CONF=0.20 python app.py        (PowerShell: $env:CONF="0.20"; python app.py)
DEFAULT_CONF = min(0.9, max(0.05, float(os.environ.get("CONF", "0.15"))))


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def label_of(name: str) -> str:
    return CLASS_INFO.get(name, {}).get("label", name.replace("_", " ").title())


def color_of(name: str) -> str:
    return CLASS_INFO.get(name, {}).get("color", "#475569")


# ---------------------------------------------------------------------------
# Struktur hasil
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    cls: str
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    area_ratio: float
    severity: int
    severity_label: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    width: int = 0
    height: int = 0
    inference_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def severity(self) -> int:
        return max((d.severity for d in self.detections), default=0)

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABELS[self.severity]


class ModelNotFoundError(RuntimeError):
    """Dilempar ketika file bobot model belum tersedia."""


# ---------------------------------------------------------------------------
# Logika tingkat keparahan
# ---------------------------------------------------------------------------
def estimate_severity(cls_name: str, area_ratio: float) -> int:
    """
    Estimasi sederhana (heuristik) berdasarkan proporsi luas kotak terhadap
    luas gambar. Lubang dan retak buaya dianggap lebih berbahaya sehingga
    naik satu tingkat bila area-nya sudah cukup berarti.
    """
    if area_ratio < 0.02:
        level = 1
    elif area_ratio < 0.08:
        level = 2
    else:
        level = 3
    if cls_name in ("pothole", "alligator_crack") and area_ratio >= 0.01:
        level = min(3, level + 1)
    return level


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class RoadDamageDetector:
    def __init__(self, model_path: str | os.PathLike | None = None, imgsz: int = 640):
        self.model_path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
        self.imgsz = imgsz
        self._model = None
        self._lock = threading.Lock()

    # -- status ------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if not self.model_path.exists():
                    raise ModelNotFoundError(
                        f"File model tidak ditemukan: {self.model_path}. "
                        "Latih model terlebih dahulu (python training/train.py) "
                        "atau letakkan best.pt di folder models/."
                    )
                from ultralytics import YOLO  # impor lambat, hanya saat dibutuhkan

                self._model = YOLO(str(self.model_path))
        return self._model

    # -- util gambar -------------------------------------------------------
    @staticmethod
    def load_image(path: str | os.PathLike, max_side: int = 2560) -> np.ndarray:
        """Baca gambar (BGR), perbaiki orientasi EXIF, dan batasi ukuran."""
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
            arr = np.asarray(im)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # -- inferensi ---------------------------------------------------------
    def predict(self, image: np.ndarray, conf: float = 0.25) -> DetectionResult:
        model = self._load()
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        results = model.predict(image, conf=conf, imgsz=self.imgsz, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        out = DetectionResult(width=w, height=h, inference_ms=round(elapsed, 1))
        names = model.names
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                name = str(names[int(k)])
                ratio = max(0.0, ((x2 - x1) * (y2 - y1)) / float(w * h))
                sev = estimate_severity(name, ratio)
                out.detections.append(
                    Detection(
                        cls=name,
                        label=label_of(name),
                        confidence=round(float(c), 4),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        area_ratio=round(ratio, 5),
                        severity=sev,
                        severity_label=SEVERITY_LABELS[sev],
                    )
                )
            out.detections.sort(key=lambda d: d.confidence, reverse=True)
        return out

    # -- anotasi -----------------------------------------------------------
    @staticmethod
    def annotate(image: np.ndarray, result: DetectionResult) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        scale = max(0.9, min(w, h) / 700)
        thick = max(2, int(round(2 * scale)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.6 * scale

        for d in result.detections:
            color = hex_to_bgr(color_of(d.cls))
            cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), color, thick, cv2.LINE_AA)

            text = f"{d.label} {d.confidence * 100:.0f}%"
            (tw, th), base = cv2.getTextSize(text, font, fscale, 1)
            pad = int(6 * scale)
            top = d.y1 - th - base - pad * 2
            if top < 0:  # label di dalam kotak bila mentok atas
                top = d.y1
            cv2.rectangle(img, (d.x1, top), (d.x1 + tw + pad * 2, top + th + base + pad * 2), color, -1)
            cv2.putText(
                img, text, (d.x1 + pad, top + th + pad),
                font, fscale, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return img

    # -- alur lengkap ------------------------------------------------------
    def process_file(
        self, src_path: str | os.PathLike, result_path: str | os.PathLike, conf: float = 0.25
    ) -> DetectionResult:
        image = self.load_image(src_path)
        result = self.predict(image, conf=conf)
        annotated = self.annotate(image, result)
        cv2.imwrite(str(result_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return result
