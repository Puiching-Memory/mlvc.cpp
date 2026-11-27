import createMlvcModule from '/dist/mlvc_webgpu.js';

const PRESETS = {
  'mlvc-s-360p': {
    modelBase: '/models/canonical/mlvc-s-psnr-v1/640x368',
    streamUrl: '/models/samples/mlvc-s-640x360.mlvc',
    fallbackStreamUrl: '/models/fixtures/references/mlvc-s-psnr-v1/gray-q21-2f/reference.mlvc',
  },
  'mlvc-s-720p': {
    modelBase: '/models/canonical/mlvc-s-psnr-v1/1280x720',
    streamUrl: '/models/samples/mlvc-s-1280x720.mlvc',
  },
  'mlvc-360p': {
    modelBase: '/models/canonical/mlvc-psnr-v1/640x368',
    streamUrl: '/models/samples/mlvc-640x360.mlvc',
    fallbackStreamUrl: '/models/fixtures/references/mlvc-psnr-v1/gray-q21-2f/reference.mlvc',
  },
};

const REQUIRED_DECODER_ASSETS = [
  'model_bundle.pb',
  'metadata.pb',
  'gaussian_pmf.pb',
  'bit_estimator_pmf.pb',
  'aot/manifest.pb',
  'aot/MLVCDecoder/graph.pb',
  'aot/MLVCDecoder/weights.bin',
];

const $ = (id) => document.getElementById(id);
const logEl = $('log');

function log(msg) {
  const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
  logEl.textContent += `[${ts}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
  console.log('[mlvc-webgpu]', msg);
}

// Splits Annex-B mlvc-nalu-v0 bitstream into per-frame Access Units.
// Every VCL NALU (types 0,1,2,3,20) ends an AU; SPS(33)/PPS(34) precede an IDR(20).
function splitAccessUnits(bytes) {
  const nalus = [];
  let i = 0;
  while (i + 4 <= bytes.length) {
    if (bytes[i] === 0 && bytes[i + 1] === 0 && bytes[i + 2] === 0 && bytes[i + 3] === 1) {
      nalus.push(i);
      i += 4;
    } else {
      i++;
    }
  }
  const aus = [];
  let auStart = -1;
  for (let k = 0; k < nalus.length; k++) {
    const start = nalus[k];
    const end = k + 1 < nalus.length ? nalus[k + 1] : bytes.length;
    if (start + 5 > end) continue;
    // 16-bit NALU header: forbidden_zero(1) | nalu_type(6) | ...
    const naluType = (bytes[start + 4] >> 1) & 0x3f;
    if (auStart < 0) auStart = start;
    const isVcl = (naluType === 0 || naluType === 1 || naluType === 2 || naluType === 3 || naluType === 20);
    if (isVcl) {
      aus.push(bytes.subarray(auStart, end));
      auStart = -1;
    }
  }
  return aus;
}

let Module = null;
let accessUnits = [];
let cursor = 0;
let playing = false;
let rafId = 0;
let metricsPtr = 0;
let auBufPtr = 0;
let auBufCap = 0;
let gpuReadyPromise = null;

function ensureAuBuffer(len) {
  if (len <= auBufCap) return;
  if (auBufPtr) Module._free(auBufPtr);
  auBufCap = Math.max(len, 262144);
  auBufPtr = Module._malloc(auBufCap);
}

async function initWasmAndGpu() {
  if (!navigator.gpu) {
    $('gpu-badge').textContent = 'WebGPU Unavailable';
    throw new Error('This browser does not expose navigator.gpu');
  }
  log('Loading mlvc_webgpu.wasm...');
  Module = await createMlvcModule();
  metricsPtr = Module._malloc(64);

  globalThis.__mlvcResizeCanvas = (sel, w, h) => {
    const el = document.querySelector(sel);
    if (el && (el.width !== w || el.height !== h)) {
      el.width = w;
      el.height = h;
    }
  };

  gpuReadyPromise = new Promise((resolve, reject) => {
    globalThis.__mlvcOnGpuReady = (ok, info) => {
      if (ok) resolve(JSON.parse(info));
      else reject(new Error(info));
    };
  });
  Module._mlvc_web_init_gpu();
  const info = await gpuReadyPromise;
  $('gpu-badge').textContent = `${info.adapter} · f16:${info.f16 ? 'ON' : 'OFF'}`;
  log(`WebGPU ready: ${JSON.stringify(info)}`);
  window.__mlvcState = { ready: true, gpuInfo: info, lastMetrics: null, benchResult: null };
}

async function loadPreset(presetKey, customStreamBytes = null) {
  playing = false;
  if (rafId) cancelAnimationFrame(rafId);
  $('btn-play').textContent = 'Play Loop';

  const preset = PRESETS[presetKey];
  log(`Fetching decoder bundle from ${preset.modelBase} ...`);
  const t0 = performance.now();

  const fetched = await Promise.all(
    REQUIRED_DECODER_ASSETS.map(async (rel) => {
      const resp = await fetch(`${preset.modelBase}/${rel}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${rel}`);
      return [rel, new Uint8Array(await resp.arrayBuffer())];
    })
  );

  Module._mlvc_web_asset_clear();
  let totalBytes = 0;
  for (const [rel, data] of fetched) {
    totalBytes += data.byteLength;
    const pData = Module._malloc(data.byteLength);
    Module.HEAPU8.set(data, pData);
    Module.ccall('mlvc_web_asset_put', null, ['string', 'number', 'number'], [rel, pData, data.byteLength]);
    Module._free(pData);
  }
  log(`Loaded ${(totalBytes / 1048576).toFixed(2)} MiB decoder assets in ${(performance.now() - t0).toFixed(0)} ms`);

  const matrix = parseInt($('matrix-select').value, 10) || 0;
  const tCompile = performance.now();
  const rc = Module.ccall('mlvc_web_decoder_create', 'number', ['string', 'number'], ['#mlvc-canvas', matrix]);
  if (rc !== 0) {
    const err = Module.UTF8ToString(Module._mlvc_web_last_error());
    throw new Error(`decoder_create failed: ${err}`);
  }
  log(`WebGPU decoder graph & WGSL pipelines ready in ${(performance.now() - tCompile).toFixed(0)} ms`);

  let streamBytes = customStreamBytes;
  if (!streamBytes) {
    let resp = await fetch(preset.streamUrl);
    if (!resp.ok && preset.fallbackStreamUrl) {
      resp = await fetch(preset.fallbackStreamUrl);
    }
    if (!resp.ok) throw new Error(`Failed to fetch sample stream (${preset.streamUrl})`);
    streamBytes = new Uint8Array(await resp.arrayBuffer());
  }

  accessUnits = splitAccessUnits(streamBytes);
  cursor = 0;
  log(`Parsed ${accessUnits.length} Access Units (${(streamBytes.byteLength / 1024).toFixed(1)} KiB total)`);

  $('btn-play').disabled = false;
  $('btn-step').disabled = false;
  $('btn-bench').disabled = false;

  // Immediately decode and render frame 0!
  decodeOneFrame();
}

function decodeOneFrame() {
  if (!accessUnits.length) return null;
  if (cursor >= accessUnits.length) {
    cursor = 0;
    Module._mlvc_web_decoder_reset();
  }
  const au = accessUnits[cursor++];
  ensureAuBuffer(au.byteLength);
  Module.HEAPU8.set(au, auBufPtr);
  const rc = Module._mlvc_web_decode_au(auBufPtr, au.byteLength, metricsPtr);
  if (rc !== 0) {
    const err = Module.UTF8ToString(Module._mlvc_web_last_error());
    log(`Decode error: ${err}`);
    playing = false;
    return null;
  }
  const i32 = Module.HEAP32;
  const f64 = Module.HEAPF64;
  const b32 = metricsPtr >> 2;
  const b64 = metricsPtr >> 3;
  const m = {
    width: i32[b32 + 0],
    height: i32[b32 + 1],
    frameIdx: i32[b32 + 2],
    qIndex: i32[b32 + 3],
    frameType: i32[b32 + 4],
    fusedD2s8: i32[b32 + 5] !== 0,
    usesF16: i32[b32 + 6] !== 0,
    dispatches: i32[b32 + 7],
    ransMs: f64[b64 + 4],
    submitMs: f64[b64 + 5],
    auBytes: au.byteLength,
  };

  $('m-rans-ms').textContent = `${m.ransMs.toFixed(2)} ms`;
  $('m-submit-ms').textContent = `${m.submitMs.toFixed(2)} ms`;
  $('i-res').textContent = `${m.width}×${m.height}`;
  $('i-frame').textContent = `#${m.frameIdx} (${m.frameType === 0 ? 'IDR' : 'P'}) · QP ${m.qIndex}`;
  $('i-dispatches').textContent = String(m.dispatches);
  $('i-f16').textContent = m.usesF16 ? 'YES (enable f16)' : 'Fallback (u32 packed)';
  $('i-d2s8').textContent = m.fusedD2s8 ? 'YES (Zero-Readback)' : 'Direct x_hat';
  $('i-bytes').textContent = `${m.auBytes} B`;
  window.__mlvcState.lastMetrics = m;
  return m;
}

let playFrames = 0;
let playStart = 0;
function playLoop() {
  if (!playing) return;
  const t0 = performance.now();
  decodeOneFrame();
  const dt = performance.now() - t0;
  playFrames++;
  const elapsed = performance.now() - playStart;
  if (elapsed >= 400) {
    const fps = (playFrames * 1000) / elapsed;
    $('m-fps').textContent = `${fps.toFixed(1)}`;
    $('m-frame-ms').textContent = `${(elapsed / playFrames).toFixed(2)} ms`;
    playFrames = 0;
    playStart = performance.now();
  }
  rafId = requestAnimationFrame(playLoop);
}

function waitGpuFlush() {
  return new Promise((resolve) => {
    globalThis.__mlvcOnGpuFlush = () => resolve();
    Module._mlvc_web_flush_gpu();
  });
}

async function runBenchmark(totalFrames = 60) {
  playing = false;
  if (rafId) cancelAnimationFrame(rafId);
  $('btn-play').textContent = 'Play Loop';

  // Warmup 5 frames and wait for GPU queue to drain cleanly.
  for (let i = 0; i < 5; i++) decodeOneFrame();
  await waitGpuFlush();

  let ransSum = 0;
  let submitSum = 0;
  const mc = new MessageChannel();
  const yieldTick = () => new Promise((r) => {
    mc.port1.onmessage = r;
    mc.port2.postMessage(null);
  });

  const t0 = performance.now();
  for (let i = 0; i < totalFrames; i++) {
    const m = decodeOneFrame();
    if (!m) break;
    ransSum += m.ransMs;
    submitSum += m.submitMs;
    // Sync every 4 frames so GPU queue stays pipelined (matching real streaming) without unbounded queue buildup.
    if ((i & 3) === 3) {
      await waitGpuFlush();
      await yieldTick();
    }
  }
  await waitGpuFlush();
  const elapsed = performance.now() - t0;
  const avgFrameMs = elapsed / totalFrames;
  const fps = 1000 / avgFrameMs;
  const avgRans = ransSum / totalFrames;
  const avgSubmit = submitSum / totalFrames;
  $('m-fps').textContent = fps.toFixed(1);
  $('m-frame-ms').textContent = `${avgFrameMs.toFixed(2)} ms`;
  const res = { totalFrames, fps, avgFrameMs, avgRans, avgSubmit };
  window.__mlvcState.benchResult = res;
  log(`Benchmark (${totalFrames}f, GPU-synced): ${fps.toFixed(1)} FPS (${avgFrameMs.toFixed(2)} ms/f, rANS=${avgRans.toFixed(2)} ms, submit=${avgSubmit.toFixed(2)} ms)`);
  return res;
}

$('btn-load').addEventListener('click', () => loadPreset($('preset-select').value).catch((e) => log(`ERR: ${e.message}`)));
$('btn-step').addEventListener('click', () => decodeOneFrame());
$('btn-play').addEventListener('click', () => {
  playing = !playing;
  $('btn-play').textContent = playing ? 'Pause' : 'Play Loop';
  if (playing) {
    playFrames = 0;
    playStart = performance.now();
    rafId = requestAnimationFrame(playLoop);
  }
});
$('btn-bench').addEventListener('click', () => runBenchmark(60));
$('file-input').addEventListener('change', async (ev) => {
  const f = ev.target.files?.[0];
  if (!f) return;
  const bytes = new Uint8Array(await f.arrayBuffer());
  log(`Selected local file: ${f.name} (${(bytes.byteLength / 1024).toFixed(1)} KiB)`);
  await loadPreset($('preset-select').value, bytes);
});

window.__mlvcLoadPreset = loadPreset;
window.__mlvcRunBenchmark = runBenchmark;

initWasmAndGpu()
  .then(() => loadPreset('mlvc-s-360p'))
  .catch((e) => log(`INIT ERROR: ${e.message}`));
