const decoderCache = new Map();

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0:00";
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function createSeededRandom(seed = 1) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

export function gaussianRandom(rand = Math.random) {
  let spare = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    const z0 = mag * Math.cos(2.0 * Math.PI * v);
    spare = mag * Math.sin(2.0 * Math.PI * v);
    return z0;
  };
}

export async function loadJSON(path, { cache = "no-store" } = {}) {
  const url = new URL(path, import.meta.url);
  const response = await fetch(url, { cache });
  if (!response.ok) {
    throw new Error(`Failed to load JSON: ${url} (${response.status})`);
  }
  return response.json();
}

export async function ensureOrtLoaded(timeoutMs = 8000) {
  if (window.ort) {
    return window.ort;
  }
  const start = performance.now();
  while (!window.ort) {
    if (performance.now() - start > timeoutMs) {
      throw new Error("onnxruntime-web failed to load within timeout");
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return window.ort;
}

function resolveAssetPath(assetPath) {
  if (assetPath.startsWith("http://") || assetPath.startsWith("https://")) {
    return assetPath;
  }
  return new URL(`../${assetPath.replace(/^\.?\//, "")}`, import.meta.url).toString();
}

function getIOName(session, kind, fallback) {
  const names = kind === "input" ? session.inputNames : session.outputNames;
  if (Array.isArray(names) && names.length > 0) {
    return names[0];
  }
  return fallback;
}

export async function createDecoderSession(meta, options = {}) {
  const ort = await ensureOrtLoaded();
  const preferInt8 = options.preferInt8 !== false;
  const providerCombosInput = options.executionProviders;
  const providerCombos = Array.isArray(providerCombosInput) && providerCombosInput.length ?
    (Array.isArray(providerCombosInput[0]) ?
      providerCombosInput.map((combo) => combo.slice()) :
      [providerCombosInput.slice()]) :
    [["webgpu"], ["wasm"]];

  const candidateModels = [];
  if (preferInt8 && meta.int8_decoder) {
    candidateModels.push({ type: "int8", url: resolveAssetPath(meta.int8_decoder) });
  }
  if (meta.fp32_decoder) {
    candidateModels.push({ type: "fp32", url: resolveAssetPath(meta.fp32_decoder) });
  }
  if (meta.fp16_decoder) {
    candidateModels.push({ type: "fp16", url: resolveAssetPath(meta.fp16_decoder) });
  }

  let lastError = null;
  for (const providers of providerCombos) {
    const cacheKey = JSON.stringify({ preferInt8, providers, order: candidateModels.map((c) => c.type) });
    if (decoderCache.has(cacheKey)) {
      return decoderCache.get(cacheKey);
    }

    for (const candidate of candidateModels) {
      try {
        const session = await ort.InferenceSession.create(candidate.url, {
          executionProviders: providers,
          graphOptimizationLevel: "all",
        });
        const info = {
          session,
          modelType: candidate.type,
          inputName: getIOName(session, "input", "latent"),
          outputName: getIOName(session, "output", "image"),
          providers,
        };
        decoderCache.set(cacheKey, info);
        return info;
      } catch (error) {
        lastError = error;
        console.warn(`[viz] Failed to load ${candidate.type} decoder with providers [${providers.join(", ")}]`, error);
      }
    }
  }

  throw new Error(`Unable to initialize decoder session${lastError ? `: ${lastError.message}` : ""}`);
}

export function calculateFps({ now, previous, frames, intervalMs = 500 }) {
  const delta = now - previous.timestamp;
  previous.accumulator += delta;
  previous.frames += frames;
  if (previous.accumulator >= intervalMs) {
    const fps = (previous.frames * 1000) / previous.accumulator;
    previous.accumulator = 0;
    previous.frames = 0;
    previous.timestamp = now;
    return fps;
  }
  previous.timestamp = now;
  return null;
}

export function takeAverage(samples) {
  if (!samples || samples.length === 0) return 0;
  const sum = samples.reduce((acc, value) => acc + value, 0);
  return sum / samples.length;
}

export function softmax(values, temperature = 1.0) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp((v - max) / Math.max(temperature, 1e-4)));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => (sum > 0 ? v / sum : 0));
}

export function mappedRange(value, inMin, inMax, outMin, outMax) {
  const t = clamp((value - inMin) / (inMax - inMin || 1), 0, 1);
  return lerp(outMin, outMax, t);
}

export function updateElementText(element, text) {
  if (!element) return;
  element.textContent = text;
}

export function toRgbaBuffer(source, { mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5] } = {}) {
  const channels = 3;
  const pixelCount = source.length / channels;
  const rgba = new Uint8ClampedArray(pixelCount * 4);
  for (let i = 0; i < pixelCount; i += 1) {
    const r = source[i * 3 + 0] * std[0] + mean[0];
    const g = source[i * 3 + 1] * std[1] + mean[1];
    const b = source[i * 3 + 2] * std[2] + mean[2];
    rgba[i * 4 + 0] = clamp(Math.round(r * 255), 0, 255);
    rgba[i * 4 + 1] = clamp(Math.round(g * 255), 0, 255);
    rgba[i * 4 + 2] = clamp(Math.round(b * 255), 0, 255);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
