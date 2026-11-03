/* ==========================================================================
   Deteksi Jalan Rusak — interaksi antarmuka
   ========================================================================== */

(function () {
  "use strict";

  // ---------- Navigasi mobile ----------
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");
  if (toggle && links) toggle.addEventListener("click", () => links.classList.toggle("open"));

  // ---------- Konfirmasi form ----------
  document.querySelectorAll("form[data-confirm]").forEach((f) => {
    f.addEventListener("submit", (e) => {
      if (!window.confirm(f.dataset.confirm)) e.preventDefault();
    });
  });

  // ---------- Flash otomatis hilang ----------
  document.querySelectorAll(".flash.success").forEach((el) => setTimeout(() => el.remove(), 4500));

  // ---------- Unggah gambar ----------
  const form = document.getElementById("uploadForm");
  if (form) {
    const input = document.getElementById("fileInput");
    const zone = document.getElementById("dropzone");
    const preview = document.getElementById("preview");
    const nameEl = document.getElementById("dzName");
    const conf = document.getElementById("conf");
    const confOut = document.getElementById("confOut");
    const overlay = document.getElementById("overlay");
    const MAX = 16 * 1024 * 1024;

    const showFile = (file) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        alert("File harus berupa gambar.");
        input.value = "";
        return;
      }
      if (file.size > MAX) {
        alert("Ukuran file terlalu besar (maksimal 16 MB).");
        input.value = "";
        return;
      }
      preview.src = URL.createObjectURL(file);
      nameEl.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
      zone.classList.add("has-file");
    };

    input.addEventListener("change", () => showFile(input.files[0]));

    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); })
    );
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); })
    );
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        showFile(input.files[0]);
      }
    });

    const syncConf = () => { confOut.textContent = Math.round(conf.value * 100) + "%"; };
    conf.addEventListener("input", syncConf);
    syncConf();

    form.addEventListener("submit", () => {
      if (input.files.length) overlay.classList.add("show");
    });
  }
})();

// ==========================================================================
// Pembanding sebelum / sesudah (halaman detail)
// ==========================================================================
function initCompare() {
  const box = document.getElementById("compare");
  if (!box) return;
  const after = document.getElementById("afterLayer");
  const handle = document.getElementById("handle");
  const tagL = document.getElementById("tagL");
  const tagR = document.getElementById("tagR");
  let pos = 50;

  const apply = () => {
    after.style.clipPath = `inset(0 0 0 ${pos}%)`;
    handle.style.left = pos + "%";
  };
  const move = (clientX) => {
    const r = box.getBoundingClientRect();
    pos = Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100));
    apply();
  };

  let drag = false;
  box.addEventListener("pointerdown", (e) => {
    if (box.dataset.mode !== "compare") return;
    drag = true; box.setPointerCapture(e.pointerId); move(e.clientX);
  });
  box.addEventListener("pointermove", (e) => { if (drag) move(e.clientX); });
  box.addEventListener("pointerup", () => { drag = false; });

  document.querySelectorAll(".view-switch button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-switch button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      const v = btn.dataset.view;
      box.dataset.mode = v;
      const compare = v === "compare";
      handle.style.display = tagL.style.display = tagR.style.display = compare ? "" : "none";
      box.style.cursor = compare ? "ew-resize" : "default";
      if (v === "result") pos = 0;
      else if (v === "original") pos = 100;
      else pos = 50;
      apply();
    });
  });
  apply();
}

// ==========================================================================
// Grafik dashboard — SVG/CSS murni (tanpa library, tetap jalan offline)
// ==========================================================================
function initDashboard(data) {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---- Garis: aktivitas harian ----
  const trend = document.getElementById("trendChart");
  if (trend) {
    const W = 640, H = 290, L = 34, R = 26, T = 14, B = 28;
    const vals = data.trend.values, labels = data.trend.labels;
    const max = Math.ceil(Math.max(4, ...vals) / 4) * 4, n = vals.length;
    const x = (i) => L + (i * (W - L - R)) / (n - 1);
    const y = (v) => T + (1 - v / max) * (H - T - B);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: "100%", preserveAspectRatio: "xMidYMid meet" }, trend);
    for (let g = 0; g <= 4; g++) {
      const gv = Math.round((max * g) / 4), gy = y(gv);
      el("line", { x1: L, x2: W - R, y1: gy, y2: gy, stroke: "rgba(21,23,28,.07)" }, svg);
      el("text", { x: L - 8, y: gy + 4, "text-anchor": "end", "font-size": 11, fill: "#6c707a" }, svg).textContent = gv;
    }
    labels.forEach((lb, i) => {
      if (i % 2 === 1 || i === n - 1)
        el("text", { x: x(i), y: H - 8, "text-anchor": i === n - 1 ? "end" : "middle", "font-size": 11, fill: "#6c707a" }, svg).textContent = lb;
    });
    const pts = vals.map((v, i) => `${x(i)},${y(v)}`);
    el("polygon", { points: `${x(0)},${y(0)} ${pts.join(" ")} ${x(n - 1)},${y(0)}`, fill: "rgba(245,158,11,.18)" }, svg);
    el("polyline", { points: pts.join(" "), fill: "none", stroke: "#15171c", "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    vals.forEach((v, i) => {
      const c = el("circle", { cx: x(i), cy: y(v), r: 4, fill: "#f59e0b", stroke: "#15171c", "stroke-width": 2 }, svg);
      el("title", {}, c).textContent = `${labels[i]}: ${v} analisis`;
    });
  }

  // ---- Donat: jenis kerusakan ----
  const cls = document.getElementById("classChart");
  if (cls) {
    const total = data.by_class.reduce((a, c) => a + c.count, 0);
    let acc = 0;
    const stops = data.by_class.map((c) => {
      const from = (acc / (total || 1)) * 100; acc += c.count;
      return `${c.color} ${from}% ${(acc / (total || 1)) * 100}%`;
    });
    cls.innerHTML =
      `<div class="donut" style="background:${total ? `conic-gradient(${stops.join(",")})` : "#efebe3"}"><div><strong>${total}</strong><span>temuan</span></div></div>` +
      `<ul class="legend">${data.by_class.map((c) =>
        `<li><i style="background:${c.color}"></i><span>${esc(c.label)}</span><b>${c.count}</b></li>`).join("")}</ul>`;
  }

  // ---- Batang: tingkat keparahan ----
  const sev = document.getElementById("sevChart");
  if (sev) {
    const colors = ["#16a34a", "#65a30d", "#d97706", "#dc2626"];
    const max = Math.max(1, ...data.by_sev.map((s) => s.count));
    sev.innerHTML = `<div class="bars">${data.by_sev.map((s, i) =>
      `<div class="bar-col"><b>${s.count}</b><div class="bar-fill" style="height:${(s.count / max) * 78}%;background:${colors[i]}"></div><span>${esc(s.label)}</span></div>`
    ).join("")}</div>`;
  }
}

// Lebar bar keyakinan (diatur lewat data-w agar template bebas CSS inline)
document.querySelectorAll("[data-w]").forEach((el) => {
  el.style.width = el.dataset.w + "%";
});

// Baris tabel yang bisa diklik (data-href) & inisialisasi dashboard dari data-stats
document.querySelectorAll("tr[data-href]").forEach((tr) => {
  tr.addEventListener("click", () => { window.location.href = tr.dataset.href; });
});
(function () {
  const holder = document.getElementById("dashData");
  if (holder) initDashboard(JSON.parse(holder.dataset.stats));
})();