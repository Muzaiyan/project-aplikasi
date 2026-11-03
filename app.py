
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from flask import (
    Flask, Response, abort, flash, g, jsonify, redirect, render_template,
    request, send_from_directory, url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from detector import (
    CLASS_INFO, SEVERITY_LABELS, ModelNotFoundError, RoadDamageDetector,
    color_of, label_of, DEFAULT_CONF,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
INSTANCE_DIR = BASE_DIR / "instance"
DB_PATH = INSTANCE_DIR / "database.db"
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
PER_PAGE = 12

for d in (UPLOAD_DIR, RESULT_DIR, INSTANCE_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-ganti-di-produksi")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

detector = RoadDamageDetector()


# ---------------------------------------------------------------------------
# Database (sqlite3 bawaan Python, tanpa dependensi tambahan)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name TEXT    NOT NULL,
    image_file    TEXT    NOT NULL,
    result_file   TEXT    NOT NULL,
    conf_threshold REAL   NOT NULL,
    total         INTEGER NOT NULL,
    severity      INTEGER NOT NULL,
    inference_ms  REAL    NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    items         TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_det_created ON detections(created_at);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(SCHEMA)


init_db()


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["boxes"] = json.loads(d["items"])
    d["severity_label"] = SEVERITY_LABELS[d["severity"]]
    d["created_dt"] = datetime.fromisoformat(d["created_at"])
    # ringkasan kelas untuk kartu riwayat
    summary: dict[str, int] = {}
    for it in d["boxes"]:
        summary[it["cls"]] = summary.get(it["cls"], 0) + 1
    d["summary"] = summary
    return d


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
@app.context_processor
def inject_globals():
    return {
        "CLASS_INFO": CLASS_INFO,
        "SEVERITY_LABELS": SEVERITY_LABELS,
        "model_ready": detector.available,
        "label_of": label_of,
        "color_of": color_of,
        "now_year": datetime.now().year,
    }


@app.template_filter("tgl")
def fmt_tgl(dt: datetime) -> str:
    bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    return f"{dt.day} {bulan[dt.month - 1]} {dt.year}, {dt:%H:%M}"


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT



def parse_conf(raw, default=DEFAULT_CONF) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return min(0.95, max(0.05, v))


def run_detection(file_storage, conf: float) -> int:
    """Simpan unggahan, jalankan YOLO, catat ke database. Mengembalikan id."""
    if not file_storage or not file_storage.filename:
        raise ValueError("Pilih gambar terlebih dahulu.")
    if not allowed(file_storage.filename):
        raise ValueError("Format tidak didukung. Gunakan JPG, PNG, BMP, atau WEBP.")

    uid = uuid.uuid4().hex
    tmp_path = UPLOAD_DIR / f"{uid}.tmp"
    file_storage.save(tmp_path)
    img_name, res_name = f"{uid}.jpg", f"{uid}_hasil.jpg"
    img_path, res_path = UPLOAD_DIR / img_name, RESULT_DIR / res_name

    try:
        try:
            image = RoadDamageDetector.load_image(tmp_path)
        except Exception as exc:  # file bukan gambar valid
            raise ValueError("File bukan gambar yang valid.") from exc
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        result = detector.process_file(img_path, res_path, conf=conf)
    except Exception:
        img_path.unlink(missing_ok=True)
        raise

    db = get_db()
    cur = db.execute(
        """INSERT INTO detections
           (original_name, image_file, result_file, conf_threshold, total, severity,
            inference_ms, width, height, items, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            Path(file_storage.filename).name[:120], img_name, res_name, conf,
            result.total, result.severity, result.inference_ms,
            result.width, result.height,
            json.dumps([d.to_dict() for d in result.detections]),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    db.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Halaman
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    rows = get_db().execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT 4"
    ).fetchall()
    return render_template("index.html", recent=[row_to_dict(r) for r in rows])


@app.route("/detect", methods=["POST"])
def detect():
    conf = parse_conf(request.form.get("conf"))
    try:
        det_id = run_detection(request.files.get("image"), conf)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except ModelNotFoundError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    except Exception as exc:  # pragma: no cover
        app.logger.exception("Deteksi gagal")
        flash(f"Terjadi kesalahan saat memproses gambar: {exc}", "error")
        return redirect(url_for("index"))
    return redirect(url_for("detail", det_id=det_id))


@app.route("/detail/<int:det_id>")
def detail(det_id: int):
    row = get_db().execute("SELECT * FROM detections WHERE id=?", (det_id,)).fetchone()
    if row is None:
        abort(404)
    return render_template("detail.html", d=row_to_dict(row))


@app.route("/history")
def history():
    q = request.args.get("q", "").strip()
    sev = request.args.get("severity", "")
    page = max(1, request.args.get("page", 1, type=int))

    where, params = [], []
    if q:
        where.append("original_name LIKE ?")
        params.append(f"%{q}%")
    if sev.isdigit() and 0 <= int(sev) <= 3:
        where.append("severity = ?")
        params.append(int(sev))
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    db = get_db()
    total = db.execute(f"SELECT COUNT(*) FROM detections {clause}", params).fetchone()[0]
    pages = max(1, -(-total // PER_PAGE))
    page = min(page, pages)
    rows = db.execute(
        f"SELECT * FROM detections {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [PER_PAGE, (page - 1) * PER_PAGE],
    ).fetchall()
    return render_template(
        "history.html",
        items=[row_to_dict(r) for r in rows],
        total=total, page=page, pages=pages, q=q, sev=sev,
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=compute_stats())


@app.route("/delete/<int:det_id>", methods=["POST"])
def delete(det_id: int):
    db = get_db()
    row = db.execute("SELECT image_file, result_file FROM detections WHERE id=?", (det_id,)).fetchone()
    if row is None:
        abort(404)
    (UPLOAD_DIR / row["image_file"]).unlink(missing_ok=True)
    (RESULT_DIR / row["result_file"]).unlink(missing_ok=True)
    db.execute("DELETE FROM detections WHERE id=?", (det_id,))
    db.commit()
    flash("Data deteksi berhasil dihapus.", "success")
    return redirect(url_for("history"))


# ---------------------------------------------------------------------------
# File statis hasil & unggahan
# ---------------------------------------------------------------------------
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.route("/download/<int:det_id>")
def download(det_id: int):
    row = get_db().execute("SELECT result_file FROM detections WHERE id=?", (det_id,)).fetchone()
    if row is None:
        abort(404)
    return send_from_directory(RESULT_DIR, row["result_file"], as_attachment=True,
                               download_name=f"hasil-deteksi-{det_id}.jpg")


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------
def compute_stats() -> dict:
    rows = [row_to_dict(r) for r in get_db().execute("SELECT * FROM detections ORDER BY id DESC").fetchall()]

    by_class = {k: 0 for k in CLASS_INFO}
    by_sev = [0, 0, 0, 0]
    total_damage = 0
    for r in rows:
        by_sev[r["severity"]] += 1
        for it in r["boxes"]:
            by_class[it["cls"]] = by_class.get(it["cls"], 0) + 1
            total_damage += 1

    # tren 14 hari terakhir
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    per_day = {d: 0 for d in days}
    for r in rows:
        dd = r["created_dt"].date()
        if dd in per_day:
            per_day[dd] += 1

    return {
        "total_scans": len(rows),
        "total_damage": total_damage,
        "avg_ms": round(sum(r["inference_ms"] for r in rows) / len(rows), 1) if rows else 0,
        "worst": by_sev[3],
        "by_class": [
            {"cls": k, "label": label_of(k), "color": color_of(k), "count": v}
            for k, v in by_class.items()
        ],
        "by_sev": [{"label": SEVERITY_LABELS[i], "count": c} for i, c in enumerate(by_sev)],
        "trend": {
            "labels": [d.strftime("%d/%m") for d in days],
            "values": [per_day[d] for d in days],
        },
        "recent": rows[:5],
    }


# ---------------------------------------------------------------------------
# API JSON
# ---------------------------------------------------------------------------
@app.route("/api/health")
def api_health():
    return jsonify(status="ok", model_ready=detector.available, model_path=str(detector.model_path.name))


@app.route("/api/stats")
def api_stats():
    s = compute_stats()
    s.pop("recent")
    return jsonify(s)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    conf = parse_conf(request.form.get("conf"))
    try:
        det_id = run_detection(request.files.get("image"), conf)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except ModelNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    row = get_db().execute("SELECT * FROM detections WHERE id=?", (det_id,)).fetchone()
    d = row_to_dict(row)
    return jsonify(
        id=det_id,
        total=d["total"],
        severity=d["severity_label"],
        inference_ms=d["inference_ms"],
        detections=d["boxes"],
        result_url=url_for("results", filename=d["result_file"], _external=True),
        detail_url=url_for("detail", det_id=det_id, _external=True),
    )


@app.route("/export.csv")
def export_csv():
    rows = [row_to_dict(r) for r in get_db().execute("SELECT * FROM detections ORDER BY id DESC").fetchall()]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "waktu", "file", "jumlah_kerusakan", "tingkat", "waktu_inferensi_ms"])
    for r in rows:
        w.writerow([r["id"], r["created_at"], r["original_name"], r["total"],
                    r["severity_label"], r["inference_ms"]])
    return Response(
        "﻿" + buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=riwayat-deteksi.csv"},
    )


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    flash("Ukuran file terlalu besar (maksimal 16 MB).", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_e):
    return render_template("base.html", not_found=True), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
