/* Unloss UI: subida -> /api/restore -> veredicto + slider 1:1 + verificación.
   Vanilla JS, sin librerías. Nunca se inserta contenido del usuario con innerHTML. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone"), fileInput = $("fileInput"), fileName = $("fileName");
  const expectedEl = $("expected"), useModelEl = $("useModel"), ocrEngineEl = $("ocrEngine");
  const runBtn = $("runBtn"), runLabel = runBtn.querySelector(".label"), errorLine = $("errorLine");
  const verdict = $("verdict"), verdictIcon = $("verdictIcon"), verdictTitle = $("verdictTitle"), verdictSub = $("verdictSub");
  const viewer = $("viewer"), emptyState = $("emptyState");
  const loadingOverlay = $("loadingOverlay"), errorOverlay = $("errorOverlay"), errorOverlayTxt = $("errorOverlayTxt");
  const compare = $("compare"), viewerTag = $("viewerTag"), viewerCaption = $("viewerCaption");
  const canvas = $("canvas"), imgUp = $("imgUp"), imgRec = $("imgRec");
  const divider = $("divider");
  const lens = $("lens"), lensUp = $("lensUp"), lensRec = $("lensRec");
  const payload = $("payloadChip"), payloadSym = $("payloadSym"), payloadText = $("payloadText"), copyBtn = $("copyBtn");
  const hud = $("hud"), hudPsnr = $("hudPsnr"), hudSsim = $("hudSsim"), hudVia = $("hudVia"), hudLat = $("hudLat");
  const methodsTable = $("methodsTable"), metricsLine = $("metricsLine"), modelLine = $("modelLine");

  let upUrl = null;
  let split = 50, dragging = false, fit = { w: 0, h: 0 }, zoom = 2.5;

  /* ---------- estados del visor ---------- */
  function setState(state, msg) {
    viewer.dataset.state = state;
    emptyState.hidden = state !== "empty";
    loadingOverlay.hidden = state !== "loading";
    errorOverlay.hidden = state !== "error";
    compare.hidden = state !== "done";
    viewerCaption.hidden = state !== "done";
    if (state === "error") errorOverlayTxt.textContent = msg || "No se pudo procesar la imagen.";
  }

  function setTag(cls, txt) {
    viewerTag.className = "viewer-tag " + (cls || "");
    viewerTag.textContent = txt;
  }

  function setVerdict(cls, icon, title, sub) {
    verdict.hidden = false;
    verdict.className = "verdict " + cls;
    verdictIcon.textContent = icon;
    verdictTitle.textContent = title;
    verdictSub.textContent = sub || "";
  }

  /* ---------- tema / tamaño / animaciones ---------- */
  const mql = window.matchMedia("(prefers-color-scheme: dark)");

  function readPref(key, valid, def) {
    try { const v = localStorage.getItem(key); return valid.includes(v) ? v : def; } catch (_) { return def; }
  }
  function writePref(key, v) { try { localStorage.setItem(key, v); } catch (_) {} }

  function applyTheme(pref) {
    const eff = pref === "light" ? "light" : pref === "dark" ? "dark" : (mql.matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", eff);
    document.querySelectorAll("#segTheme .seg-btn").forEach((b) => {
      const on = b.dataset.themePref === pref;
      b.setAttribute("aria-checked", String(on));
    });
  }
  function applySize(size) {
    if (size) document.documentElement.setAttribute("data-size", size);
    else document.documentElement.removeAttribute("data-size");
    document.querySelectorAll("#segSize .seg-btn").forEach((b) => {
      const on = (b.dataset.sizeVal || "") === size;
      b.setAttribute("aria-checked", String(on));
    });
  }

  const themePref = readPref("unloss_theme", ["system", "light", "dark"], "system");
  applyTheme(themePref);
  applySize(readPref("unloss_textsize", ["lg", "xl"], ""));
  const motionPref = readPref("unloss_motion", ["reduce"], "");
  if (motionPref === "reduce") document.documentElement.setAttribute("data-motion", "reduce");
  const reduceMotion = $("reduceMotion");
  reduceMotion.checked = motionPref === "reduce";
  reduceMotion.addEventListener("change", () => {
    if (reduceMotion.checked) {
      document.documentElement.setAttribute("data-motion", "reduce");
      writePref("unloss_motion", "reduce");
    } else {
      document.documentElement.removeAttribute("data-motion");
      try { localStorage.removeItem("unloss_motion"); } catch (_) {}
    }
  });

  document.querySelectorAll("#segTheme .seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      const pref = b.dataset.themePref;
      writePref("unloss_theme", pref);
      applyTheme(pref);
    });
  });
  document.querySelectorAll("#segSize .seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      const size = b.dataset.sizeVal || "";
      if (size) writePref("unloss_textsize", size);
      else { try { localStorage.removeItem("unloss_textsize"); } catch (_) {} }
      applySize(size);
    });
  });
  mql.addEventListener("change", () => {
    if (readPref("unloss_theme", ["system", "light", "dark"], "system") === "system") applyTheme("system");
  });

  /* panel de accesibilidad */
  const a11yBtn = $("a11yBtn"), a11yPanel = $("a11yPanel");
  function toggleA11y(open) {
    a11yPanel.hidden = !open;
    a11yBtn.setAttribute("aria-expanded", String(open));
  }
  a11yBtn.addEventListener("click", () => toggleA11y(a11yPanel.hidden));
  document.addEventListener("click", (e) => {
    if (!a11yPanel.hidden && !a11yBtn.contains(e.target) && !a11yPanel.contains(e.target)) toggleA11y(false);
  });

  /* ---------- subida ---------- */
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

  function setFile(f) {
    if (upUrl) URL.revokeObjectURL(upUrl);
    upUrl = URL.createObjectURL(f);
    fileName.hidden = false;
    fileName.textContent = f.name + " · " + (f.size / (1024 * 1024)).toFixed(2) + " MB";
    const dzPreview = $("dzPreview");
    if (dzPreview) {
      dzPreview.src = upUrl;
      dzPreview.hidden = false;
      dropzone.classList.add("has-file");
    }
    errorLine.hidden = true;
  }

  /* ---------- ejecución ---------- */
  runBtn.addEventListener("click", run);

  async function run() {
    errorLine.hidden = true;
    if (!fileInput.files[0]) {
      errorLine.textContent = "Primero sube una imagen.";
      errorLine.hidden = false;
      return;
    }
    setLoading(true);
    const fd = new FormData();
    fd.append("image", fileInput.files[0]);
    const exp = expectedEl.value.trim();
    if (exp) fd.append("expected", exp);
    if (useModelEl.checked) fd.append("use_model", "true");
    fd.append("ocr_engine", ocrEngineEl.value);
    try {
      const r = await fetch("/api/restore", { method: "POST", body: fd });
      let j;
      const text = await r.text();
      try {
        j = JSON.parse(text);
      } catch (err) {
        if (!r.ok) throw new Error(`Error del servidor (Código ${r.status}). Es posible que la imagen sea muy grande y el proceso se haya reiniciado por falta de memoria.`);
        throw new Error("Respuesta inválida del servidor: " + text.slice(0, 50));
      }
      if (!r.ok) throw new Error(j.detail || "error del servidor");
      render(j);
    } catch (err) {
      errorLine.textContent = "No se pudo procesar: " + err.message;
      errorLine.hidden = false;
      setState("error", "No se pudo procesar: " + err.message);
      setVerdict("err", "×", "Error al procesar la imagen", err.message);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(on) {
    runBtn.disabled = on;
    runLabel.textContent = on ? "Procesando…" : "Restaurar y verificar";
    if (on) {
      const spin = document.createElement("span");
      spin.className = "spin";
      spin.setAttribute("aria-hidden", "true");
      runBtn.prepend(spin);
      setState("loading");
    } else {
      const spin = runBtn.querySelector(".spin");
      if (spin) spin.remove();
    }
  }

  /* ---------- render ---------- */
  function render(j) {
    setState("done");
    const d = j.domain === "text" ? "text" : "qr";
    const v = V[j.status] || V.err;
    const eff = j.note ? "warn" : v.cls; /* lectura parecida: avisar, no marcar ok */
    setVerdict(
      eff,
      v.icon,
      v.title[d],
      (j.note ? j.note + " · " : "") + (V[j.status] ? v.sub[d] : v.sub[d])
        + (j.elapsed_ms ? " · " + (j.elapsed_ms / 1000).toFixed(2) + " s" : "")
    );
    setTag(eff, v.tagTxt[d] || "×");
    if (j.note) setTag("warn", "REVISA !");

    imgUp.src = upUrl;
    imgRec.src = j.reconstruction.data_url;
    lensUp.style.backgroundImage = "url(" + upUrl + ")";
    lensRec.style.backgroundImage = "url(" + j.reconstruction.data_url + ")";

    const onLoad = () => {
      const nw = imgUp.naturalWidth || imgRec.naturalWidth;
      const nh = imgUp.naturalHeight || imgRec.naturalHeight;
      const maxW = Math.min(viewer.clientWidth - 52, 720);
      const maxH = 520;
      const s = Math.min(maxW / nw, maxH / nh, 3); // Allow scaling up to 3x for tiny images
      fit = { w: Math.round(nw * s), h: Math.round(nh * s) };
      canvas.style.width = fit.w + "px";
      canvas.style.height = fit.h + "px";
      setSplit(50);
      lens.hidden = true;
      canvas.classList.toggle("scan", j.status === "verified");
    };
    if (imgUp.complete && imgUp.naturalWidth) onLoad();
    else imgUp.addEventListener("load", onLoad, { once: true });

    if (j.status === "verified" || j.status === "decoded" || j.status === "false_positive") {
      payload.hidden = false;
      payloadSym.className = "sym " + (j.note || j.status === "false_positive" ? "warn" : "ok");
      payloadSym.textContent = (j.note || j.status === "false_positive") ? "!" : "✓";
      payloadText.textContent = j.decoded_payload || "";
      copyBtn.hidden = false;
    } else {
      payload.hidden = false;
      payloadSym.className = "sym err";
      payloadSym.textContent = "×";
      payloadText.textContent = "No se pudo decodificar";
      copyBtn.hidden = true;
    }

    // HUD (mini-stats)
    const mt = j.metrics || {};
    hud.hidden = false;
    hudPsnr.textContent = mt.subida_vs_rec_psnr != null ? Number(mt.subida_vs_rec_psnr).toFixed(2) + " dB" : "—";
    hudSsim.textContent = mt.subida_vs_rec_ssim != null ? Number(mt.subida_vs_rec_ssim).toFixed(3) : "—";
    const via = (j.methods || []).find((m) => m.best) || (j.methods || []).find((m) => m.decoded);
    hudVia.textContent = via ? (via.engine && via.engine !== "tesseract" ? via.name + " · " + via.engine : via.name) : "—";
    hudVia.className = "hud-val" + (via ? " ok" : "");
    const lat = (j.elapsed_ms / 1000).toFixed(2) + " s";
    hudLat.textContent = lat;
    hudLat.className = "hud-val" + (j.note ? " warn" : (j.status === "verified" ? " ok" : (j.status === "false_positive" ? " warn" : "")));

    // tabla de métodos (sin innerHTML)
    methodsTable.replaceChildren();
    const head = methodsTable.createTHead();
    const hr = head.insertRow();
    ["Método", "Lector", "Decodifica", "Payload", "ms"].forEach((t) => {
      const th = document.createElement("th");
      th.textContent = t;
      hr.appendChild(th);
    });
    const tb = methodsTable.createTBody();
    for (const m of j.methods) {
      const tr = tb.insertRow();
      const tdName = document.createElement("td"); tdName.className = "name"; tdName.textContent = m.name;
      const tdEng = document.createElement("td");
      tdEng.textContent = m.domain === "text"
        ? (m.engine === "rapid" ? "RapidOCR" : "Tesseract")
        : "cv2";
      if (m.engine === "rapid") tdEng.className = "accent";
      const tdDec = document.createElement("td"); tdDec.className = m.decoded ? "ok" : "no"; tdDec.textContent = m.decoded ? "✓" : "×";
      const tdPay = document.createElement("td"); tdPay.textContent = m.payload || "—";
      const tdT = document.createElement("td"); tdT.className = "t"; tdT.textContent = String(m.elapsed_ms);
      [tdName, tdEng, tdDec, tdPay, tdT].forEach((td) => tr.appendChild(td));
    }
    if (j.methods.length < 9) {
      const tr = tb.insertRow();
      const td = document.createElement("td");
      td.colSpan = 5;
      td.className = "no";
      if (j.selection === "high_conf") {
        td.textContent = "Detenido al lograr una lectura con confianza alta (acota latencia).";
      } else if (j.selection === "exact") {
        td.textContent = "Detenido al lograr una lectura que coincide con el contenido esperado.";
      } else if (j.selection === "best") {
        td.textContent = "Se probaron todos los métodos; se muestra la mejor lectura.";
      } else {
        td.textContent = "Detenido tras el primer método que decodificó (acota latencia).";
      }
      tr.appendChild(td);
    }

    metricsLine.textContent = "PSNR subida vs reconstrucción: " + (mt.subida_vs_rec_psnr ?? "—")
      + " · SSIM: " + (mt.subida_vs_rec_ssim ?? "—") + " (métricas de píxel sobre vista reducida: no predicen si decodifica)";
    const mdl = j.model || {};
    modelLine.textContent = mdl.attempted
      ? ("Ruta del modelo v9b (experimental): " + (mdl.payload ? "decodificó " + mdl.payload : "no decodificó") + " — " + (mdl.note || "") + (mdl.reason ? " (" + mdl.reason + ")" : ""))
      : "Ruta del modelo v9b: desactivada (" + (mdl.reason || "no solicitada") + ")";

    viewer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  const V = {
    verified: {
      cls: "ok", icon: "✓",
      title: { qr: "Decodificó y coincide con el contenido esperado", text: "El texto leído coincide con el contenido esperado" },
      sub: { qr: "Verificado por el pipeline clásico", text: "Comparación con normalización OCR (LineAcc)" },
      tagTxt: { qr: "ESCANEADO ✓", text: "TEXTO LEÍDO ✓" },
    },
    decoded: {
      cls: "ok", icon: "✓",
      title: { qr: "Decodificó correctamente", text: "El OCR leyó texto en la imagen" },
      sub: { qr: "Sin contenido esperado para comparar", text: "Sin contenido esperado para comparar" },
      tagTxt: { qr: "DECODIFICADO", text: "TEXTO LEÍDO" },
    },
    false_positive: {
      cls: "warn", icon: "!",
      title: { qr: "Decodificó pero NO coincide con lo esperado", text: "El OCR leyó texto pero NO coincide con lo esperado" },
      sub: { qr: "Revisa el contenido antes de usarlo", text: "Revisa el texto leído antes de usarlo" },
      tagTxt: { qr: "NO COINCIDE !", text: "NO COINCIDE !" },
    },
    not_decoded: {
      cls: "err", icon: "×",
      title: { qr: "No se pudo decodificar", text: "No se pudo leer (ni QR ni texto)" },
      sub: { qr: "El pipeline solo decodifica códigos QR y texto (OCR). Si tu imagen es otra cosa, no hay canal de lectura.", text: "Se probaron QR y texto (Tesseract). Si tu imagen es otra cosa, no hay canal de lectura." },
      tagTxt: { qr: "×", text: "×" },
    },
    err: {
      cls: "err", icon: "×",
      title: { qr: "Resultado desconocido", text: "Resultado desconocido" },
      sub: { qr: "", text: "" },
      tagTxt: { qr: "×", text: "×" },
    },
  };

  /* ---------- slider ---------- */
  function setSplit(p) {
    split = Math.max(0, Math.min(100, p));
    divider.style.left = split + "%";
    imgUp.style.clipPath = "inset(0 " + (100 - split) + "% 0 0)";
    divider.setAttribute("aria-valuenow", String(Math.round(split)));
  }

  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    moveSplit(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (dragging) moveSplit(e);
    updateLens(e);
  });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointerleave", () => { lens.hidden = true; });

  function moveSplit(e) {
    const r = canvas.getBoundingClientRect();
    setSplit(((e.clientX - r.left) / r.width) * 100);
  }

  /* ---------- lupa comparativa ---------- */
  function updateLens(e) {
    if (!fit.w) return;
    const r = canvas.getBoundingClientRect();
    const cx = e.clientX - r.left, cy = e.clientY - r.top;
    const LW = 180, LH = 200;
    lens.hidden = false;
    let lx = cx + 26, ly = cy + 26;
    if (lx + LW > r.width) lx = cx - LW - 26;
    if (ly + LH > r.height) ly = cy - LH - 26;
    lens.style.left = lx + "px";
    lens.style.top = ly + "px";
    const bs = (fit.w * zoom) + "px " + (fit.h * zoom) + "px";
    const bp = (-(cx * zoom - LW / 2)) + "px " + (-(cy * zoom - LH / 2)) + "px";
    lensUp.style.backgroundSize = bs; lensRec.style.backgroundSize = bs;
    lensUp.style.backgroundPosition = bp; lensRec.style.backgroundPosition = bp;
  }

  /* ---------- copiar payload ---------- */
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(payloadText.textContent);
      copyBtn.textContent = "Copiado ✓";
      setTimeout(() => { copyBtn.textContent = "Copiar"; }, 1500);
    } catch (_) {
      copyBtn.textContent = "No se pudo copiar";
    }
  });

  /* ---------- aviso inicial + cierre ---------- */
  const modal = $("modal");
  if (!localStorage.getItem("unloss_accept")) modal.hidden = false;
  $("modalOk").addEventListener("click", () => {
    localStorage.setItem("unloss_accept", "1");
    modal.hidden = true;
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!modal.hidden) modal.hidden = true;
      if (!a11yPanel.hidden) toggleA11y(false);
    }
  });
})();