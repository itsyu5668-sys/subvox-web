/* Subvox Web — plain JS, no build step.
   Calls the FastAPI backend. Every result shown comes from a real API
   response; missing fields are shown as "not reported", never invented.
   No audio is ever captured: getUserMedia is called with audio: false. */
'use strict';

/* ================= API base ================= */
function resolveApiBase() {
  const q = new URLSearchParams(location.search).get('api');
  if (q) return q.replace(/\/$/, '');
  if (window.SUBVOX_API_BASE) return String(window.SUBVOX_API_BASE).replace(/\/$/, '');
  try {
    const saved = localStorage.getItem('subvox_api_base');
    if (saved) return saved.replace(/\/$/, '');
  } catch (e) { /* storage unavailable, fall through */ }
  return '/api';
}
let API_BASE = resolveApiBase();

async function apiJson(path, opts) {
  const res = await fetch(API_BASE + path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = { _raw: text }; }
  if (!res.ok) {
    const msg = (data && (data.detail || data.error || data.message)) || ('HTTP ' + res.status + ' ' + res.statusText);
    throw new Error(msg + ' — ' + path);
  }
  return data;
}
function apiErrorText(err) {
  if (err instanceof TypeError) return 'Could not reach the API at ' + API_BASE + '. Is the backend running? (' + err.message + ')';
  return String(err.message || err);
}

/* ================= Helpers ================= */
const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function fmtMs(ms) {
  if (ms == null || isNaN(ms)) return 'not reported';
  if (ms < 1000) return Math.round(ms) + ' ms';
  return (ms / 1000).toFixed(1) + ' s';
}
function fmtPct(x) {
  if (x == null || isNaN(x)) return '—';
  return (x <= 1 ? x * 100 : x).toFixed(1) + '%';
}
function fmtTime(ts) {
  try { return new Date(ts).toLocaleString(); } catch (e) { return String(ts); }
}
/* Word error rate, computed locally in the browser. Used only for the
   optional "paste a target" comparison and as a labelled fallback when
   the server does not report WER. */
function wordErrorRate(reference, hypothesis) {
  const norm = (s) => (s || '').toLowerCase().replace(/[^a-z0-9' ]/g, '').split(/\s+/).filter(Boolean);
  const r = norm(reference), h = norm(hypothesis);
  if (!r.length) return null;
  const d = Array.from({ length: r.length + 1 }, (_, i) => [i, ...Array(h.length).fill(0)]);
  for (let j = 0; j <= h.length; j++) d[0][j] = j;
  for (let i = 1; i <= r.length; i++)
    for (let j = 1; j <= h.length; j++)
      d[i][j] = Math.min(
        d[i - 1][j] + 1,
        d[i][j - 1] + 1,
        d[i - 1][j - 1] + (r[i - 1] === h[j - 1] ? 0 : 1));
  return d[r.length][h.length] / r.length;
}

/* ================= Harvard sentences (exact, from subvox-local sentences.py) ================= */
const FAST_10 = [
  'The birch canoe slid on the smooth planks',
  'Glue the sheet to the dark blue background',
  "It's easy to tell the depth of a well",
  'These days a chicken leg is a rare dish',
  'Rice is often served in round bowls',
  'The juice of lemons makes fine punch',
  'The box was thrown beside the parked truck',
  'Four hours of steady work faced us',
  'Kick the ball straight and follow through',
  'The salt breeze came across from the sea',
];
const QUICK_6_IDX = [0, 2, 4, 5, 8, 9]; // stage 1
const STAGE2_IDX = [1, 3, 6, 7];          // stage 2
const TEST_SENTENCE = FAST_10[0];
const TAKES_PER_SENTENCE = 3;

/* ================= State ================= */
const state = {
  stream: null,
  recorder: null,
  recording: null, // {chunks, sentenceIdx, take, kind, startedAt}
  takes: {},       // "sentenceIdx:take" -> Blob
  currentSentence: 0,
  currentTake: 0,
  trained: false,
  adaptedKept: null,
  testRun: 0,
};
function takeKey(s, t) { return s + ':' + t; }
function takesRecorded() { return Object.keys(state.takes).length; }
function totalTakes() { return FAST_10.length * TAKES_PER_SENTENCE; }

/* ================= Tabs ================= */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => showScreen(tab.dataset.screen));
});
function showScreen(name) {
  document.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.screen === name)));
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === 'screen-' + name));
  if (name === 'doctor') loadDoctor();
  if (name === 'test' || name === 'dictate') attachSharedVideo(name);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ================= Health check ================= */
async function checkHealth() {
  const pill = $('apiStatus'), text = $('apiStatusText');
  try {
    const data = await apiJson('/health');
    pill.classList.add('ok'); pill.classList.remove('bad');
    text.textContent = 'API online' + (data && data.model ? ' · ' + data.model : '');
  } catch (err) {
    pill.classList.add('bad'); pill.classList.remove('ok');
    text.textContent = 'API unreachable — check Models tab';
  }
}

/* ================= Camera + mouth crop ================= */
const video = $('camVideo');
const mouthCanvas = $('mouthCanvas');
const mouthCtx = mouthCanvas.getContext('2d');
let cropRaf = null;

function drawMouthCrop() {
  if (!state.stream || video.readyState < 2) { cropRaf = requestAnimationFrame(drawMouthCrop); return; }
  const vw = video.videoWidth, vh = video.videoHeight;
  // Fixed lower-centre crop: mouth sits roughly x 30–70%, y 60–88% of frame.
  const sx = vw * 0.30, sy = vh * 0.60, sw = vw * 0.40, sh = vh * 0.28;
  mouthCtx.fillStyle = '#160f0c';
  mouthCtx.fillRect(0, 0, 96, 96);
  try { mouthCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 96, 96); } catch (e) { /* frame not ready */ }
  cropRaf = requestAnimationFrame(drawMouthCrop);
}

async function startCamera() {
  const notice = $('camNotice');
  notice.textContent = '';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false, // Subvox never records audio.
    });
    state.stream = stream;
    video.srcObject = stream;
    video.style.display = 'block';
    $('cameraOff').style.display = 'none';
    $('mouthGuide').style.display = 'block';
    $('btnStartCam').disabled = true;
    $('btnStopCam').disabled = false;
    $('btnRecordTake').disabled = false;
    $('btnRecordTest').disabled = false;
    $('btnHoldTalk').disabled = false;
    attachSharedVideo('test');
    attachSharedVideo('dictate');
    if (cropRaf) cancelAnimationFrame(cropRaf);
    cropRaf = requestAnimationFrame(drawMouthCrop);
    notice.textContent = 'Camera on. Video only — no audio is captured.';
  } catch (err) {
    notice.className = 'notice error';
    notice.textContent = 'Camera could not be started: ' + err.name + ' — ' + (err.message || 'permission denied or no camera. Training, testing and dictation all need the camera.');
  }
}
function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  if (cropRaf) cancelAnimationFrame(cropRaf);
  video.srcObject = null;
  video.style.display = 'none';
  $('cameraOff').style.display = '';
  $('mouthGuide').style.display = 'none';
  $('btnStartCam').disabled = false;
  $('btnStopCam').disabled = true;
  $('btnRecordTake').disabled = true;
  $('btnRecordTest').disabled = true;
  $('btnHoldTalk').disabled = true;
  mouthCtx.fillStyle = '#160f0c';
  mouthCtx.fillRect(0, 0, 96, 96);
}
function attachSharedVideo(screen) {
  if (!state.stream) return;
  const v = screen === 'test' ? $('testVideo') : $('dictateVideo');
  const off = screen === 'test' ? $('testCameraOff') : $('dictateCameraOff');
  v.srcObject = state.stream;
  v.style.display = 'block';
  off.style.display = 'none';
}
$('btnStartCam').addEventListener('click', startCamera);
$('btnStopCam').addEventListener('click', stopCamera);

/* ================= Recording (MediaRecorder, webm) ================= */
let recTimerInt = null;
function startRecording(kind, sentenceIdx, take) {
  if (!state.stream) return false;
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9'
    : (MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : '');
  const rec = new MediaRecorder(state.stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.start(100);
  state.recorder = rec;
  state.recording = { chunks, sentenceIdx, take, kind, startedAt: Date.now() };
  $('recIndicator').classList.add('on');
  const started = Date.now();
  recTimerInt = setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    $('recTimer').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
  }, 250);
  return true;
}
function stopRecording() {
  return new Promise((resolve) => {
    const rec = state.recorder;
    const ctx = state.recording;
    if (!rec || !ctx) { resolve(null); return; }
    clearInterval(recTimerInt);
    $('recIndicator').classList.remove('on');
    rec.onstop = () => {
      const blob = new Blob(ctx.chunks, { type: rec.mimeType || 'video/webm' });
      state.recorder = null;
      state.recording = null;
      resolve({ blob, ...ctx, durationMs: Date.now() - ctx.startedAt });
    };
    try { rec.stop(); } catch (e) { resolve(null); }
  });
}
/* Hold-to-record helper: press and hold, release to stop. */
function bindHold(button, onStart, onStop) {
  let holding = false;
  const begin = (e) => {
    e.preventDefault();
    if (holding || button.disabled) return;
    holding = true;
    button.classList.add('holding');
    if (!onStart()) { holding = false; button.classList.remove('holding'); }
  };
  const end = async (e) => {
    if (e) e.preventDefault();
    if (!holding) return;
    holding = false;
    button.classList.remove('holding');
    await onStop();
  };
  button.addEventListener('pointerdown', begin);
  button.addEventListener('pointerup', end);
  button.addEventListener('pointerleave', end);
  button.addEventListener('pointercancel', end);
  button.addEventListener('contextmenu', (e) => e.preventDefault());
  // Keyboard: hold Space while focused.
  button.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.repeat) { e.preventDefault(); begin(e); }
  });
  button.addEventListener('keyup', (e) => {
    if (e.code === 'Space') { e.preventDefault(); end(e); }
  });
}

/* ================= Train screen ================= */
function renderSentenceList() {
  const list = $('sentenceList');
  list.innerHTML = '';
  FAST_10.forEach((s, i) => {
    const done = [0, 1, 2].every((t) => state.takes[takeKey(i, t)]);
    const count = [0, 1, 2].filter((t) => state.takes[takeKey(i, t)]).length;
    const item = el('div', 'sentence-item' + (done ? ' done' : '') + (i === state.currentSentence ? ' current' : ''));
    const stage = QUICK_6_IDX.includes(i) ? 'Stage 1' : 'Stage 2';
    item.appendChild(el('span', null, (i + 1) + '. ' + s));
    item.appendChild(el('span', 'takes', count + '/3 · ' + stage));
    item.addEventListener('click', () => { state.currentSentence = i; renderTrain(); });
    list.appendChild(item);
  });
}
function renderTrain() {
  $('currentSentence').textContent = FAST_10[state.currentSentence];
  const stage = QUICK_6_IDX.includes(state.currentSentence) ? 'Stage 1' : 'Stage 2';
  $('sentenceCount').textContent =
    'Sentence ' + (state.currentSentence + 1) + ' of 10 · ' + stage +
    (QUICK_6_IDX.includes(state.currentSentence) ? ' (quick adapt)' : ' (background)');
  $('trainStageLabel').textContent = stage === 'Stage 1' ? 'Stage 1 of 2 · Quick adapt' : 'Stage 2 of 2 · Background';
  document.querySelectorAll('#takeDots .take-dot').forEach((d) => {
    const t = Number(d.dataset.take);
    d.classList.toggle('done', !!state.takes[takeKey(state.currentSentence, t)]);
    d.classList.toggle('recording', t === state.currentTake);
  });
  $('trainProgressFill').style.width = (takesRecorded() / totalTakes() * 100) + '%';
  $('trainProgressText').textContent = takesRecorded() + ' of ' + totalTakes() + ' takes recorded';
  $('btnUploadTrain').disabled = takesRecorded() < totalTakes();
  renderSentenceList();
}
document.querySelectorAll('#takeDots .take-dot').forEach((d) => {
  d.addEventListener('click', () => { state.currentTake = Number(d.dataset.take); renderTrain(); });
});
$('btnPrevSentence').addEventListener('click', () => {
  state.currentSentence = (state.currentSentence + FAST_10.length - 1) % FAST_10.length;
  renderTrain();
});
$('btnNextSentence').addEventListener('click', () => {
  state.currentSentence = (state.currentSentence + 1) % FAST_10.length;
  renderTrain();
});
$('btnClearTakes').addEventListener('click', () => {
  state.takes = {};
  $('trainNotice').textContent = '';
  $('trainResult').style.display = 'none';
  renderTrain();
});

bindHold(
  $('btnRecordTake'),
  () => {
    if (!state.stream) { $('trainNotice').className = 'notice error'; $('trainNotice').textContent = 'Start the camera first.'; return false; }
    $('trainNotice').textContent = 'Recording take ' + (state.currentTake + 1) + ' — mouth the sentence silently, then release.';
    return startRecording('train', state.currentSentence, state.currentTake);
  },
  async () => {
    const result = await stopRecording();
    if (!result) return;
    if (result.blob.size < 1000) {
      $('trainNotice').className = 'notice error';
      $('trainNotice').textContent = 'That take came back almost empty (' + result.blob.size + ' bytes). Try again, holding a little longer.';
      return;
    }
    state.takes[takeKey(result.sentenceIdx, result.take)] = result.blob;
    $('trainNotice').className = 'notice';
    $('trainNotice').textContent = 'Saved take ' + (result.take + 1) + ' of sentence ' + (result.sentenceIdx + 1) +
      ' (' + (result.blob.size / 1024).toFixed(0) + ' KB, ' + (result.durationMs / 1000).toFixed(1) + ' s).';
    // Advance: next take, or next sentence.
    if (result.take < TAKES_PER_SENTENCE - 1) state.currentTake = result.take + 1;
    else if (result.sentenceIdx < FAST_10.length - 1) { state.currentSentence = result.sentenceIdx + 1; state.currentTake = 0; }
    renderTrain();
  }
);

$('btnUploadTrain').addEventListener('click', async () => {
  const notice = $('trainNotice');
  const btn = $('btnUploadTrain');
  btn.disabled = true;
  $('trainJob').style.display = 'block';
  $('jobProgressText').textContent = 'Uploading 30 takes…';
  $('jobLossText').textContent = '';
  try {
    const form = new FormData();
    const manifest = [];
    FAST_10.forEach((sentence, i) => {
      for (let t = 0; t < TAKES_PER_SENTENCE; t++) {
        const blob = state.takes[takeKey(i, t)];
        const field = 'video_' + i + '_' + t;
        form.append(field, blob, 'sentence' + i + '_take' + t + '.webm');
        manifest.push({ sentence, sentence_index: i, take: t, stage: QUICK_6_IDX.includes(i) ? 1 : 2, field });
      }
    });
    form.append('manifest', JSON.stringify(manifest));
    form.append('stage', '1');
    const data = await apiJson('/train', { method: 'POST', body: form });
    await handleTrainResult(data);
  } catch (err) {
    notice.className = 'notice error';
    notice.textContent = 'Training upload failed: ' + apiErrorText(err) + ' No takes were lost — they are still stored in this page, try again.';
    $('jobProgressText').textContent = 'Training did not start.';
  } finally {
    btn.disabled = takesRecorded() < totalTakes();
  }
});

async function handleTrainResult(data) {
  // The server may answer immediately with results, or with a job id to poll.
  if (data && data.job_id && !('stock_wer' in data) && !('adapted_wer' in data)) {
    $('jobProgressText').textContent = 'Training job ' + data.job_id + ' started. Polling for progress…';
    for (let i = 0; i < 120; i++) { // up to ~10 min at 5 s
      await new Promise((r) => setTimeout(r, 5000));
      try {
        const job = await apiJson('/train/' + encodeURIComponent(data.job_id));
        const pct = job.progress != null ? job.progress : (job.step != null && job.total_steps ? job.step / job.total_steps * 100 : null);
        if (pct != null) $('jobProgressFill').style.width = pct + '%';
        $('jobProgressText').textContent =
          (job.status || 'training') +
          (job.step != null ? ' — step ' + job.step + (job.total_steps ? ' of ' + job.total_steps : '') : '') +
          (job.stage ? ' · ' + job.stage : '');
        if (job.loss != null) $('jobLossText').textContent = 'loss: ' + Number(job.loss).toFixed(4);
        if (job.status === 'done' || job.status === 'complete' || job.status === 'finished' || 'stock_wer' in job || 'adapted_wer' in job) {
          showTrainResult(job);
          return;
        }
        if (job.status === 'failed' || job.status === 'error') {
          throw new Error(job.error || job.detail || 'training job failed on the server');
        }
      } catch (err) {
        // If the poll endpoint does not exist, say so honestly instead of looping forever.
        if (/404/.test(String(err.message))) throw new Error('The server accepted the upload but has no /train/{job_id} status endpoint to poll. Training may still be running — check the Models tab, then re-check.');
        throw err;
      }
    }
    throw new Error('Training is still running after 10 minutes of polling. It may finish in the background — check the Models tab later.');
  }
  showTrainResult(data);
}

function showTrainResult(data) {
  const notice = $('trainNotice');
  $('jobProgressFill').style.width = '100%';
  $('jobProgressText').textContent = 'Training finished.';
  if (data && data.loss != null) $('jobLossText').textContent = 'final loss: ' + Number(data.loss).toFixed(4);
  const stock = data && (data.stock_wer ?? data.stockWer ?? data.baseline_wer);
  const adapted = data && (data.adapted_wer ?? data.adaptedWer ?? data.finetuned_wer);
  $('trainResult').style.display = 'block';
  $('werStock').textContent = fmtPct(stock);
  $('werAdapted').textContent = fmtPct(adapted);
  if (stock != null && adapted != null) {
    const delta = (stock - adapted) * 100;
    $('werDelta').textContent = (delta >= 0 ? '−' : '+') + Math.abs(delta).toFixed(1) + ' pts';
    $('werDeltaSub').textContent = delta >= 0 ? 'adapted is better' : 'adapted is worse — consider discarding';
  } else {
    $('werDelta').textContent = '—';
    $('werDeltaSub').textContent = 'server did not report both WERs';
  }
  const kept = data && (data.kept ?? data.keep_adapted ?? data.adapted_kept);
  state.adaptedKept = kept == null ? null : !!kept;
  $('werKept').textContent = kept == null ? 'not reported' : (kept ? 'Yes' : 'No');
  if (kept == null) {
    notice.className = 'notice';
    notice.textContent = 'Training finished, but the server did not say whether the adapted model was kept. Use the buttons below to decide — the choice is sent to the server.';
  } else {
    notice.className = 'notice';
    notice.textContent = kept
      ? 'Adapted model kept by the server. Dictation is now unlocked.'
      : 'Adapted model was NOT kept (it did not beat stock on held-out sentences). Dictation still works with the stock model.';
  }
  state.trained = true;
  $('chipTrain').classList.add('done');
  $('chipTrainState').textContent = 'done';
  $('chipDictateState').textContent = 'ready';
  document.querySelector('[data-screen="test"]').focus();
}

async function sendKeepDecision(keep) {
  const notice = $('keepNotice');
  notice.textContent = 'Sending your decision…';
  try {
    const data = await apiJson('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: keep ? 'keep' : 'discard' }),
    });
    state.adaptedKept = keep;
    $('werKept').textContent = keep ? 'Yes' : 'No';
    notice.className = 'notice';
    notice.textContent = keep
      ? 'Adapted model kept' + (data && data.message ? ' — ' + data.message : '.')
      : 'Adapted model discarded — stock model is active' + (data && data.message ? ' — ' + data.message : '.');
  } catch (err) {
    notice.className = 'notice error';
    notice.textContent = 'The server did not accept the decision: ' + apiErrorText(err) +
      ' The choice was NOT saved server-side. It is recorded here only as your intent: ' + (keep ? 'keep adapted' : 'discard adapted') + '.';
    state.adaptedKept = keep; // local intent only
    $('werKept').textContent = (keep ? 'Yes' : 'No') + ' (local only)';
  }
}
$('btnKeepAdapted').addEventListener('click', () => sendKeepDecision(true));
$('btnDiscardAdapted').addEventListener('click', () => sendKeepDecision(false));

/* ================= Test screen ================= */
let testTakes = {}; // take -> {raw, cleaned, wer, werStock, werAdapted, latencyMs}
function renderTestDots() {
  document.querySelectorAll('#testTakeDots .take-dot').forEach((d) => {
    d.classList.toggle('done', !!testTakes[Number(d.dataset.take)]);
  });
}
document.querySelectorAll('#testTakeDots .take-dot').forEach((d) => {
  d.addEventListener('click', () => { state.testTake = Number(d.dataset.take); });
});
state.testTake = 0;

bindHold(
  $('btnRecordTest'),
  () => {
    if (!state.stream) { $('testNotice').className = 'notice error'; $('testNotice').textContent = 'Start the camera on the Train screen first.'; return false; }
    $('testNotice').textContent = 'Recording test take ' + (state.testTake + 1) + ' — mouth the sentence, then release.';
    return startRecording('test', 0, state.testTake);
  },
  async () => {
    const result = await stopRecording();
    if (!result) return;
    const notice = $('testNotice');
    notice.className = 'notice';
    notice.textContent = 'Scoring take ' + (result.take + 1) + '…';
    try {
      const form = new FormData();
      form.append('video', result.blob, 'test_take' + result.take + '.webm');
      form.append('sentence', TEST_SENTENCE);
      form.append('take', String(result.take));
      const started = Date.now();
      const data = await apiJson('/test', { method: 'POST', body: form });
      const latencyMs = data.latency_ms ?? data.latencyMs ?? (Date.now() - started);
      const raw = data.raw ?? data.raw_text ?? data.transcript ?? data.text ?? null;
      const cleaned = data.cleaned ?? data.cleaned_text ?? null;
      let wer = data.wer ?? data.wer_adapted ?? data.adapted_wer ?? null;
      let werStock = data.wer_stock ?? data.stock_wer ?? null;
      let werAdapted = data.wer_adapted ?? data.adapted_wer ?? null;
      // Labelled local fallback only if the server did not report WER.
      let werNote = '';
      if (wer == null && raw != null) {
        wer = wordErrorRate(TEST_SENTENCE, raw);
        werNote = ' (computed locally in your browser — server did not report WER)';
      }
      testTakes[result.take] = { raw, cleaned, wer, werStock, werAdapted, latencyMs };
      renderTestDots();
      renderTestRun(result.take, testTakes[result.take], werNote);
      appendTestLog({ sentence: TEST_SENTENCE, werStock, werAdapted: werAdapted ?? wer, latencyMs, raw });
      notice.textContent = 'Take ' + (result.take + 1) + ' scored. WER ' + fmtPct(wer) + werNote;
      if (result.take < 2) state.testTake = result.take + 1;
      $('chipTest').classList.add('done');
      $('chipTestState').textContent = 'done';
    } catch (err) {
      notice.className = 'notice error';
      notice.textContent = 'Test scoring failed: ' + apiErrorText(err) + ' The take was recorded but not scored — nothing was logged.';
    }
  }
);

function renderTestRun(take, r, werNote) {
  const wrap = $('testRunResults');
  const card = el('div', 'panel');
  card.appendChild(el('h3', null, 'Take ' + (take + 1)));
  const kv = el('dl', 'kv');
  const add = (k, v) => { kv.appendChild(el('dt', null, k)); kv.appendChild(el('dd', null, v)); };
  add('Raw output', r.raw ?? 'not reported');
  add('Cleaned', r.cleaned ?? 'not reported');
  add('WER', fmtPct(r.wer) + (werNote || ''));
  add('Stock WER', fmtPct(r.werStock));
  add('Adapted WER', fmtPct(r.werAdapted ?? r.wer));
  add('Latency', fmtMs(r.latencyMs));
  card.appendChild(kv);
  wrap.prepend(card);
}

/* Test log, persisted in this browser only. */
function loadTestLog() {
  try { return JSON.parse(localStorage.getItem('subvox_test_log') || '[]'); } catch (e) { return []; }
}
function saveTestLog(log) {
  try { localStorage.setItem('subvox_test_log', JSON.stringify(log.slice(0, 200))); } catch (e) { /* ignore */ }
}
function renderTestLog() {
  const log = loadTestLog();
  const body = $('testLogBody');
  body.innerHTML = '';
  if (!log.length) {
    body.appendChild(el('tr')).innerHTML = '<td colspan="7" class="muted">No tests yet. Run one above.</td>';
    return;
  }
  log.forEach((row) => {
    const tr = el('tr');
    [fmtTime(row.date), row.sentence, row.model || 'not reported',
     fmtPct(row.werStock), fmtPct(row.werAdapted), fmtMs(row.latencyMs),
     row.raw || 'not reported'].forEach((v) => tr.appendChild(el('td', null, v)));
    body.appendChild(tr);
  });
}
function appendTestLog(row) {
  const log = loadTestLog();
  log.unshift({ date: new Date().toISOString(), model: state.adaptedKept ? 'adapted' : 'stock', ...row });
  saveTestLog(log);
  renderTestLog();
}
$('btnClearLog').addEventListener('click', () => { saveTestLog([]); renderTestLog(); });

/* ================= Dictate screen ================= */
let transcriptLines = [];
bindHold(
  $('btnHoldTalk'),
  () => {
    if (!state.stream) { $('dictateNotice').className = 'notice error'; $('dictateNotice').textContent = 'Start the camera on the Train screen first.'; return false; }
    $('dictateNotice').textContent = 'Listening — mouth your sentence, then release.';
    return startRecording('dictate', -1, 0);
  },
  async () => {
    const result = await stopRecording();
    if (!result) return;
    const notice = $('dictateNotice');
    notice.className = 'notice';
    notice.textContent = 'Transcribing…';
    try {
      const form = new FormData();
      form.append('video', result.blob, 'dictate.webm');
      const started = Date.now();
      const data = await apiJson('/transcribe', { method: 'POST', body: form });
      const latencyMs = data.latency_ms ?? data.latencyMs ?? (Date.now() - started);
      const raw = data.raw ?? data.raw_text ?? data.text ?? null;
      const cleaned = data.cleaned ?? data.cleaned_text ?? null;
      const text = cleaned || raw;
      if (!text) {
        notice.className = 'notice error';
        notice.textContent = 'The server answered but returned no transcript. Raw response had no raw/cleaned/text field — see Models tab for the API shape.';
      } else {
        transcriptLines.push(text);
        const t = $('liveTranscript');
        t.innerHTML = '';
        transcriptLines.forEach((line) => t.appendChild(el('p', null, line)));
        notice.textContent = 'Transcribed in ' + fmtMs(latencyMs) + '.';
      }
      $('lastRun').style.display = 'block';
      $('lastRaw').textContent = raw ?? 'not reported';
      $('lastCleaned').textContent = cleaned ?? 'not reported (no language-model cleanup returned)';
      $('lastLatency').textContent = fmtMs(latencyMs);
      const hyps = $('hypotheses');
      hyps.innerHTML = '';
      const list = data.hypotheses || data.top_hypotheses || [];
      if (list.length) {
        list.slice(0, 3).forEach((h, i) => {
          const textH = typeof h === 'string' ? h : (h.text || h.transcript || JSON.stringify(h));
          const score = typeof h === 'object' && h.score != null ? 'score ' + Number(h.score).toFixed(3) : '';
          const row = el('div', 'hypo', (i + 1) + '. ' + textH);
          if (score) row.appendChild(el('span', 'score', score));
          hyps.appendChild(row);
        });
      } else {
        hyps.appendChild(el('p', 'small muted', 'Server returned no hypotheses list — only the top transcript above.'));
      }
      updateTargetWer(text);
    } catch (err) {
      notice.className = 'notice error';
      notice.textContent = 'Transcription failed: ' + apiErrorText(err) + ' Nothing was added to the transcript.';
    }
  }
);
function updateTargetWer(text) {
  const target = $('targetText').value.trim();
  const out = $('targetWer');
  if (!target || !text) { out.textContent = ''; return; }
  const wer = wordErrorRate(target, text);
  out.textContent = wer == null ? '' : 'Local WER vs target: ' + fmtPct(wer) + ' (computed in your browser)';
}
$('targetText').addEventListener('input', () => updateTargetWer(transcriptLines[transcriptLines.length - 1] || ''));
$('btnClearTarget').addEventListener('click', () => { $('targetText').value = ''; $('targetWer').textContent = ''; });

/* ================= Doctor / models ================= */
async function loadDoctor() {
  const notice = $('doctorNotice');
  notice.className = 'notice';
  notice.textContent = 'Asking the server…';
  try {
    const data = await apiJson('/doctor');
    notice.textContent = '';
    $('doctorContent').style.display = 'block';
    const kv = $('doctorKv');
    kv.innerHTML = '';
    const pick = (obj, keys) => { for (const k of keys) if (obj && obj[k] != null) return obj[k]; return undefined; };
    const rows = [
      ['Status', pick(data, ['status', 'ok', 'healthy']) ?? 'not reported'],
      ['Device', pick(data, ['device', 'compute_device']) ?? 'not reported'],
      ['Backend', pick(data, ['backend', 'engine']) ?? 'not reported'],
      ['Stock model', pick(data, ['stock_model', 'model', 'model_name']) ?? 'not reported'],
      ['Adapted model', pick(data, ['adapted_model', 'adapted', 'finetuned_model']) ?? 'not reported'],
      ['Adapted kept?', (data && data.adapted_kept != null) ? (data.adapted_kept ? 'Yes' : 'No') : 'not reported'],
      ['Uptime', data && data.uptime != null ? String(data.uptime) : 'not reported'],
      ['Version', pick(data, ['version', 'api_version']) ?? 'not reported'],
    ];
    rows.forEach(([k, v]) => {
      kv.appendChild(el('dt', null, k));
      kv.appendChild(el('dd', null, String(v)));
    });
    const models = data.models || data.model_list || [];
    const body = $('modelTableBody');
    body.innerHTML = '';
    if (!models.length) {
      const tr = el('tr');
      tr.appendChild(el('td', null, 'No model list returned'));
      const td = el('td', 'muted', 'the server did not include a models list in /doctor');
      td.colSpan = 3;
      tr.appendChild(td);
      body.appendChild(tr);
    } else {
      models.forEach((m) => {
        const tr = el('tr');
        const name = typeof m === 'string' ? m : (m.name || m.id || 'unnamed model');
        const size = typeof m === 'object' ? (m.size || m.size_mb || m.bytes || m.size_human || 'not reported') : 'not reported';
        const status = typeof m === 'object' ? (m.status || m.state || (m.loaded ? 'loaded' : 'not reported')) : 'not reported';
        const notes = typeof m === 'object' ? (m.notes || m.path || '') : '';
        tr.appendChild(el('td', null, name));
        tr.appendChild(el('td', null, typeof size === 'number' ? (size > 1e6 ? (size / 1e6).toFixed(0) + ' MB' : size + ' bytes') : String(size)));
        tr.appendChild(el('td', null, String(status)));
        tr.appendChild(el('td', 'muted', String(notes)));
        body.appendChild(tr);
      });
    }
    $('doctorDump').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    notice.className = 'notice error';
    notice.textContent = 'Doctor check failed: ' + apiErrorText(err) + ' No model data is shown — the numbers above would be guesses.';
    $('doctorContent').style.display = 'none';
  }
}
$('btnRefreshDoctor').addEventListener('click', loadDoctor);

/* Language-model training */
$('btnTrainLm').addEventListener('click', async () => {
  const notice = $('lmNotice');
  const sentences = $('lmText').value.split('\n').map((s) => s.trim()).filter(Boolean);
  if (!sentences.length) {
    notice.className = 'notice error';
    notice.textContent = 'Paste at least one sentence first.';
    return;
  }
  notice.className = 'notice';
  notice.textContent = 'Training language model on ' + sentences.length + ' sentence(s)…';
  try {
    const data = await apiJson('/train-lm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentences }),
    });
    notice.textContent = 'Language model trained.' + (data && data.message ? ' ' + data.message : '') +
      (data && data.perplexity != null ? ' Perplexity: ' + Number(data.perplexity).toFixed(2) + '.' : '');
  } catch (err) {
    notice.className = 'notice error';
    notice.textContent = 'Language-model training failed: ' + apiErrorText(err);
  }
});

/* ================= API base setting ================= */
$('apiBaseInput').value = API_BASE;
$('btnSaveApiBase').addEventListener('click', () => {
  const v = $('apiBaseInput').value.trim().replace(/\/$/, '') || '/api';
  API_BASE = v;
  try { localStorage.setItem('subvox_api_base', v); } catch (e) { /* ignore */ }
  checkHealth();
  loadDoctor();
});

/* ================= Init ================= */
renderTrain();
renderTestLog();
checkHealth();
