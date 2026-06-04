"use strict";

const API = {
  status:       "/api/status",
  faces:        "/api/faces",
  register:     "/api/register",
  registerSnap: "/api/register/snapshot",
  registerAuto: "/api/register/auto",
  faceQuality:  "/api/face/quality",
  cameraInfo:   "/api/camera",
  cameraSwitch: "/api/camera/switch",
};

const EMOTION_EMOJI = {
  happy:"😊", angry:"😠", sad:"😢",
  surprise:"😲", fear:"😨", disgust:"🤢", neutral:"😐",
};
const EMOTION_COLOR = {
  happy:"#00FF88", angry:"#FF3333", sad:"#3388FF",
  surprise:"#FFAA00", fear:"#AA00FF", disgust:"#88FF00", neutral:"#aaa",
};

// ── Toast ─────────────────────────────────────────────────────────
function toast(msg, type = "info", dur = 3500) {
  const el = document.createElement("div");
  el.className = `toast ${type}`; el.textContent = msg;
  document.getElementById("toast-container").appendChild(el);
  setTimeout(() => el.remove(), dur);
}

// ── Utils ─────────────────────────────────────────────────────────
function setDot(id, active) {
  document.getElementById(id)?.classList.toggle("active", !!active);
}
function setBar(id, pct) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.width = `${Math.round(pct * 100)}%`;
  el.style.background = pct >= 0.6 ? "var(--success)"
                      : pct >= 0.35 ? "var(--warning)"
                      : "var(--error)";
}

// ── Camera card ───────────────────────────────────────────────────
function updateCameraUI(info) {
  const badge  = document.getElementById("cam-source-badge");
  const status = document.getElementById("cam-status");
  const res    = document.getElementById("cam-resolution");
  if (!info || !badge) return;
  const type = info.type ?? "unknown";
  badge.textContent = type.toUpperCase();
  badge.className   = `cam-badge ${type}`;
  badge.title       = info.source ?? "";
  if (res) res.textContent = info.resolution ?? "--";
  if (status) {
    status.textContent = info.running ? "Connected" : "Disconnected";
    status.style.color = info.running ? "var(--success)" : "var(--error)";
  }
}

async function pollCamera() {
  try {
    const r = await fetch(API.cameraInfo);
    if (r.ok) updateCameraUI(await r.json());
  } catch (_) {}
}

function initCameraForm() {
  const form = document.getElementById("camera-form");
  const btn  = document.getElementById("cam-switch-btn");
  const inp  = document.getElementById("cam-source-input");
  if (!form) return;
  form.addEventListener("submit", async e => {
    e.preventDefault();
    const src = inp.value.trim();
    if (!src) { toast("Enter index or RTSP URL", "error"); return; }
    btn.disabled = true; btn.textContent = "...";
    try {
      const r = await fetch(API.cameraSwitch, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({source:src}),
      });
      const d = await r.json();
      if (r.ok && d.success) {
        toast(`Camera: ${d.type} — ${d.source}`, "success");
        updateCameraUI(d); inp.value = "";
        document.getElementById("stream").src = "/video_feed?" + Date.now();
      } else toast(d.error ?? "Switch failed", "error");
    } catch(e) { toast("Network error", "error"); }
    finally { btn.disabled=false; btn.textContent="Switch"; }
  });
}

// ── Stats ─────────────────────────────────────────────────────────
let uptimeBase = null;
async function pollStatus() {
  try {
    const r = await fetch(API.status);
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById("fps").textContent        = d.fps ?? "--";
    document.getElementById("face-count").textContent = d.face_count ?? "--";
    if (uptimeBase === null) uptimeBase = Date.now() - (d.uptime ?? 0)*1000;
    const up = Math.floor((Date.now()-uptimeBase)/1000);
    document.getElementById("uptime-badge").textContent =
      `Uptime: ${Math.floor(up/3600)}h ${Math.floor((up%3600)/60)}m ${up%60}s`;
    setDot("mod-detection",      d.modules?.detection);
    setDot("mod-recognition",    d.modules?.recognition);
    setDot("mod-emotion",        d.modules?.emotion);
    setDot("mod-reconstruction", d.modules?.reconstruction);
    const emo = d.dominant_emotion;
    const el  = document.getElementById("dominant-emotion");
    if (el) {
      el.textContent = emo ? `${EMOTION_EMOJI[emo]??""} ${emo}` : "--";
      el.style.borderColor = emo ? (EMOTION_COLOR[emo]??"") : "";
      el.style.color       = emo ? (EMOTION_COLOR[emo]??"") : "";
    }
    if (d.camera) updateCameraUI(d.camera);
  } catch(_) {}
}

// ── Face quality meter ────────────────────────────────────────────
let _qualityInterval = null;
function startQualityPolling() {
  if (_qualityInterval) return;
  _qualityInterval = setInterval(pollQuality, 800);
  pollQuality();
}
function stopQualityPolling() {
  if (_qualityInterval) { clearInterval(_qualityInterval); _qualityInterval=null; }
}
async function pollQuality() {
  try {
    const r = await fetch(API.faceQuality);
    if (!r.ok) return;
    const d = await r.json();
    const ql = document.getElementById("qlabel");
    if (!d.face) {
      if (ql) ql.textContent = "no face";
      setBar("q-sharp",0); setBar("q-bright",0); setBar("q-contrast",0);
      return;
    }
    const q = d.quality;
    if (ql) ql.textContent = `${d.label} (${Math.round(q.overall*100)}%)`;
    setBar("q-sharp",   q.sharpness);
    setBar("q-bright",  q.brightness);
    setBar("q-contrast",q.contrast);
  } catch(_) {}
}

// ── Faces list ────────────────────────────────────────────────────
async function pollFaces() {
  try {
    const r = await fetch(API.faces);
    if (!r.ok) return;
    const d = await r.json();
    const names = d.faces ?? [];
    const stats = d.stats ?? {};
    document.getElementById("known-count").textContent = names.length;
    const ul = document.getElementById("faces-list");
    if (!names.length) {
      ul.innerHTML = '<li class="empty">No faces registered yet</li>';
    } else {
      ul.innerHTML = names.map(n => {
        const cnt = stats[n] ?? 0;
        return `<li>
          <div class="face-info">
            <span class="face-name">${n}</span>
            <span class="face-count">${cnt} embedding${cnt!==1?"s":""}</span>
          </div>
          <button class="del-btn" data-name="${encodeURIComponent(n)}" title="Delete ${n}">✕</button>
        </li>`;
      }).join("");
      ul.querySelectorAll(".del-btn").forEach(b =>
        b.addEventListener("click", () => deleteFace(decodeURIComponent(b.dataset.name)))
      );
    }
  } catch(_) {}
}

async function deleteFace(name) {
  if (!confirm(`Delete all embeddings for "${name}"?`)) return;
  try {
    const r = await fetch(`${API.faces}/${encodeURIComponent(name)}`, {method:"DELETE"});
    const d = await r.json();
    if (r.ok && d.success) {
      toast(`"${name}" deleted (${d.deleted} embeddings)`, "success");
      pollFaces();
    } else toast(d.error ?? "Delete failed", "error");
  } catch(e) { toast("Network error", "error"); }
}

// ── Tabs ──────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      const t = tab.dataset.tab;
      document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab===t));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.toggle("active", c.dataset.tab===t));
      if (t === "auto") startQualityPolling();
      else stopQualityPolling();
    });
  });
}

// ── Auto Capture ──────────────────────────────────────────────────
let _selectedCount = 5;
let _jobId         = null;
let _jobPoll       = null;

function initAutoCapture() {
  // Count buttons
  document.querySelectorAll(".count-btn").forEach(b => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".count-btn").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      _selectedCount = parseInt(b.dataset.count);
    });
  });

  document.getElementById("auto-start-btn").addEventListener("click", startAutoCapture);
  document.getElementById("auto-cancel-btn").addEventListener("click", cancelAutoCapture);

  // Start quality polling for the default active tab
  startQualityPolling();
}

async function startAutoCapture() {
  const name = document.getElementById("auto-name").value.trim();
  if (!name) { toast("Enter a name first", "error"); return; }

  const startBtn  = document.getElementById("auto-start-btn");
  const cancelBtn = document.getElementById("auto-cancel-btn");
  const pbox      = document.getElementById("progress-box");
  const fill      = document.getElementById("progress-fill");
  const plabel    = document.getElementById("progress-label");
  const pmsg      = document.getElementById("progress-msg");

  startBtn.disabled   = true;
  cancelBtn.hidden    = false;
  pbox.hidden         = false;
  fill.style.width    = "0%";
  plabel.textContent  = `0 / ${_selectedCount}`;
  pmsg.textContent    = "Starting...";

  try {
    const r = await fetch(API.registerAuto, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({name, count:_selectedCount}),
    });
    const d = await r.json();
    if (!r.ok) { toast(d.error ?? "Failed to start", "error"); resetAutoUI(); return; }
    _jobId = d.job_id;
    _jobPoll = setInterval(() => pollAutoJob(name), 600);
  } catch(e) {
    toast("Network error: " + e.message, "error");
    resetAutoUI();
  }
}

async function pollAutoJob(name) {
  if (!_jobId) return;
  try {
    const r = await fetch(`${API.registerAuto}/${_jobId}`);
    if (!r.ok) return;
    const job = await r.json();

    const fill   = document.getElementById("progress-fill");
    const plabel = document.getElementById("progress-label");
    const pmsg   = document.getElementById("progress-msg");

    const pct = job.total > 0 ? job.progress / job.total : 0;
    fill.style.width   = `${Math.round(pct*100)}%`;
    plabel.textContent = `${job.progress} / ${job.total} frames`;
    pmsg.textContent   = job.message ?? "";

    if (job.status === "done") {
      clearInterval(_jobPoll); _jobPoll = null;
      toast(`"${name}" registered! ${job.saved} embeddings saved`, "success", 5000);
      resetAutoUI();
      pollFaces();
    } else if (job.status === "failed") {
      clearInterval(_jobPoll); _jobPoll = null;
      toast(job.message ?? "Auto capture failed", "error");
      resetAutoUI();
    }
  } catch(_) {}
}

function cancelAutoCapture() {
  if (_jobPoll) { clearInterval(_jobPoll); _jobPoll = null; }
  _jobId = null;
  toast("Capture cancelled", "info");
  resetAutoUI();
}

function resetAutoUI() {
  _jobId = null;
  document.getElementById("auto-start-btn").disabled = false;
  document.getElementById("auto-cancel-btn").hidden  = true;
  // keep progress visible so user can see final state
}

// ── Upload Photo ──────────────────────────────────────────────────
function initRegisterForm() {
  const imgInput = document.getElementById("reg-image");
  const preview  = document.getElementById("preview-img");
  const wrapper  = document.getElementById("preview-wrapper");
  const label    = document.querySelector(".file-label");
  const form     = document.getElementById("register-form");
  const btn      = form.querySelector("button[type='submit']");

  imgInput.addEventListener("change", () => {
    const file = imgInput.files[0]; if (!file) return;
    label.textContent = file.name;
    const reader = new FileReader();
    reader.onload = e => { preview.src = e.target.result; wrapper.hidden = false; };
    reader.readAsDataURL(file);
  });

  form.addEventListener("submit", async e => {
    e.preventDefault();
    const name = document.getElementById("reg-name").value.trim();
    const file = imgInput.files[0];
    if (!name || !file) { toast("Enter name and choose image", "error"); return; }
    btn.disabled = true; btn.textContent = "Registering...";
    const fd = new FormData(); fd.append("name",name); fd.append("image",file);
    try {
      const r = await fetch(API.register, {method:"POST", body:fd});
      const d = await r.json();
      if (r.ok && d.success) {
        toast(`"${name}" registered (${d.embeddings} embeddings)`, "success");
        form.reset(); label.textContent = "Choose Photo"; wrapper.hidden = true;
        pollFaces();
      } else toast(d.error ?? "Registration failed", "error");
    } catch(e) { toast("Network error", "error"); }
    finally { btn.disabled=false; btn.textContent="Register + Augment"; }
  });
}

// ── Snapshot ──────────────────────────────────────────────────────
function initSnapshotForm() {
  const btn = document.getElementById("snap-btn");
  const inp = document.getElementById("snap-name");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const name = inp.value.trim();
    if (!name) { toast("Enter a name first", "error"); return; }
    btn.disabled = true; btn.textContent = "Capturing...";
    try {
      const r = await fetch(API.registerSnap, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({name}),
      });
      const d = await r.json();
      if (r.ok && d.success) {
        toast(`"${name}" registered (${d.embeddings} embeddings)`, "success");
        inp.value = ""; pollFaces();
      } else toast(d.error ?? "Failed", "error");
    } catch(e) { toast("Network error", "error"); }
    finally { btn.disabled=false; btn.textContent="Capture Frame"; }
  });
}

// ── Init ──────────────────────────────────────────────────────────
function init() {
  initTabs();
  initCameraForm();
  initAutoCapture();
  initRegisterForm();
  initSnapshotForm();
  pollCamera(); pollStatus(); pollFaces();
  setInterval(pollStatus, 1000);
  setInterval(pollFaces,  6000);
  setInterval(pollCamera, 5000);
}
document.addEventListener("DOMContentLoaded", init);
