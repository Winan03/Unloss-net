/* Unloss UI: subida -> /api/restore -> slider antes/después con zoom + verificación.
   Vanilla JS, sin librerías. Nunca se inserta contenido del usuario con innerHTML. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone"), fileInput = $("fileInput"), fileName = $("fileName");
  const expectedEl = $("expected"), useModelEl = $("useModel");
  const runBtn = $("runBtn"), errorLine = $("errorLine");
  const result = $("result"), banner = $("statusBanner");
  const canvas = $("canvas"), imgUp = $("imgUp"), imgRec = $("imgRec");
  const divider = $("divider"), scanline = $("scanline");
  const lens = $("lens"), lensUp = $("lensUp"), lensRec = $("lensRec");
  const overlayTag = $("overlayTag");
  const payload = $("payloadChip"), payloadSym = $("payloadSym"), payloadText = $("payloadText"), copyBtn = $("copyBtn");
  const methodsTable = $("methodsTable"), metricsLine = $("metricsLine"), modelLine = $("modelLine");

  const STATUS = {
    verified:      { cls: "ok",   text: "✓ Decodificó y coincide con el contenido esperado." },
    decoded:       { cls: "ok",   text: "✓ Decodificó. Ver el payload en el chip." },
    false_positive:{ cls: "warn", text: "! Decodificó, pero NO coincide con lo esperado. Revisa antes de usar." },
    not_decoded:   { cls: "err",  text: "× No se pudo decodificar con el pipeline clásico." },
    error:         { cls: "err",  text: "No se pudo procesar la imagen." },
  };

  let upUrl = null;
  let split = 50, dragging = false, fit = { w: 0, h: 0 }, zoom = 2.5;

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
    fileName.textContent = f.name + " · " + (f.size / (1024 * 1024)).toFixed(2) + " MB";
    errorLine.hidden = true;
  }

  /* ---------- ejecución ---------- */
  runBtn.addEventListener("click", run);

  async function run() {
    errorLine.hidden = true;
    if (!fileInput.files[0]) { errorLine.textContent = "Primero sube una imagen."; errorLine.hidden = false; return; }
    runBtn.disabled = true;
    runBtn.textContent = "Procesando… (la primera llamada puede tardar por el arranque del servidor)";
    const fd = new FormData();
    fd.append("image", fileInput.files[0]);
    const exp = expectedEl.value.trim();
    if (exp) fd.append("expected", exp);
    if (useModelEl.checked) fd.append("use_model", "true");
    try {
      const r = await fetch("/api/restore", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "error del servidor");
      render(j);
    } catch (err) {
      errorLine.textContent = "No se pudo procesar: " + err.message;
      errorLine.hidden = false;
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "Restaurar y verificar";
    }
  }

  /* ---------- render ---------- */
  function render(j) {
    const st = STATUS[j.status] || STATUS.error;
    banner.className = "banner " + st.cls;
    banner.textContent = st.text;

    imgUp.src = upUrl;
    imgRec.src = j.reconstruction.data_url;
    lensUp.style.backgroundImage = "url(" + upUrl + ")";
    lensRec.style.backgroundImage = "url(" + j.reconstruction.data_url + ")";

    // el lienzo adopta el aspecto de la subida (reconstrucción tiene el mismo aspecto)
    const onLoad = () => {
      const nw = imgUp.naturalWidth || imgRec.naturalWidth;
      const nh = imgUp.naturalHeight || imgRec.naturalHeight;
      const maxW = Math.min(window.innerWidth * 0.9, 720);
      const maxH = 520;
      const s = Math.min(maxW / nw, maxH / nh, 1);
      fit = { w: Math.round(nw * s), h: Math.round(nh * s) };
      canvas.style.width = fit.w + "px";
      canvas.style.height = fit.h + "px";
      setSplit(50);
      lens.hidden = true;
      if (j.status === "verified") canvas.classList.add("scan"); else canvas.classList.remove("scan");
    };
    if (imgUp.complete && imgUp.naturalWidth) onLoad();
    else imgUp.addEventListener("load", onLoad, { once: true });

    // chip de payload
    if (j.status === "verified" || j.status === "decoded" || j.status === "false_positive") {
      payload.hidden = false;
      payloadSym.className = "sym " + (j.status === "false_positive" ? "warn" : "ok");
      payloadSym.textContent = j.status === "false_positive" ? "!" : "✓";
      payloadText.textContent = j.decoded_payload || "";
      copyBtn.hidden = false;
    } else {
      payload.hidden = false;
      payloadSym.className = "sym err";
      payloadSym.textContent = "×";
      payloadText.textContent = "No se pudo decodificar";
      copyBtn.hidden = true;
    }

    // overlay sobre el lienzo
    overlayTag.hidden = false;
    overlayTag.className = "overlay-tag " + st.cls;
    overlayTag.textContent = j.status === "verified" ? "ESCANEADO ✓"
      : (j.status === "false_positive" ? "NO COINCIDE !"
         : (j.status === "not_decoded" ? "×" : ""));

    // tabla de métodos (sin innerHTML)
    methodsTable.replaceChildren();
    const head = methodsTable.createTHead();
    const hr = head.insertRow();
    ["Método", "Decodifica", "Payload", "ms"].forEach((t) => {
      const th = document.createElement("th");
      th.textContent = t;
      hr.appendChild(th);
    });
    const tb = methodsTable.createTBody();
    for (const m of j.methods) {
      const tr = tb.insertRow();
      const tdName = document.createElement("td"); tdName.className = "name"; tdName.textContent = m.name;
      const tdDec = document.createElement("td"); tdDec.className = m.decoded ? "ok" : "no"; tdDec.textContent = m.decoded ? "✓" : "×";
      const tdPay = document.createElement("td"); tdPay.textContent = m.payload || "—";
      const tdT = document.createElement("td"); tdT.className = "t"; tdT.textContent = String(m.elapsed_ms);
      [tdName, tdDec, tdPay, tdT].forEach((td) => tr.appendChild(td));
    }
    if (j.methods.length < 9) {
      const tr = tb.insertRow();
      const td = document.createElement("td");
      td.colSpan = 4;
      td.className = "no";
      td.textContent = "Detenido tras el primer método que decodificó (acota latencia).";
      tr.appendChild(td);
    }

    const mt = j.metrics || {};
    metricsLine.textContent = "PSNR subida vs reconstrucción: " + (mt.subida_vs_rec_psnr ?? "—")
      + " · SSIM: " + (mt.subida_vs_rec_ssim ?? "—") + " (métricas de píxel sobre vista reducida: no predicen si decodifica)";
    const mdl = j.model || {};
    modelLine.textContent = mdl.attempted
      ? ("Ruta del modelo v9b (experimental): " + (mdl.payload ? "decodificó " + mdl.payload : "no decodificó") + " — " + (mdl.note || "") + (mdl.reason ? " (" + mdl.reason + ")" : ""))
      : "Ruta del modelo v9b: desactivada (" + (mdl.reason || "no solicitada") + ")";

    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

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

  /* ---------- aviso inicial ---------- */
  const modal = $("modal");
  if (!localStorage.getItem("unloss_accept")) modal.hidden = false;
  $("modalOk").addEventListener("click", () => {
    localStorage.setItem("unloss_accept", "1");
    modal.hidden = true;
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) modal.hidden = true;
  });
})();