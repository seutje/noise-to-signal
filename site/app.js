import { createAudioPipeline, bindFileInputToAudio, createTestTone } from "./audio.js";
import { createFeatureExtractor } from "./features.js";
import { createController } from "./controller.js";
import { createCanvasViz } from "./viz.canvas.js";
import { createThreeViz } from "./viz.three.js";
import { createRecorder } from "./recorder.js";
import {
  calculateFps,
  createDecoderSession,
  formatTime,
  loadJSON,
  updateElementText,
} from "./utils.js";

const elements = {
  stage2d: document.getElementById("stage2d"),
  threeRoot: document.getElementById("three-root"),
  playBtn: document.getElementById("playBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  seek: document.getElementById("seek"),
  time: document.getElementById("time"),
  quality: document.getElementById("quality"),
  renderer: document.getElementById("renderer"),
  sensitivity: document.getElementById("sensitivity"),
  smoothing: document.getElementById("smoothing"),
  recordBtn: document.getElementById("recordBtn"),
  fps: document.getElementById("fps"),
  statusMessage: document.getElementById("statusMessage"),
  messages: document.getElementById("messages"),
  nowPlayingTitle: document.getElementById("title"),
  art: document.getElementById("art"),
  fileInput: document.getElementById("fileInput"),
  testToneBtn: document.getElementById("testToneBtn"),
  audio: document.getElementById("albumAudio"),
};

const STORAGE_KEYS = {
  index: "nts_idx",
  time: "nts_t",
};

function postMessage(text, level = "info") {
  if (elements.messages) {
    elements.messages.textContent = text;
    elements.messages.dataset.level = level;
  }
  if (elements.statusMessage) {
    elements.statusMessage.textContent = text;
  }
}

async function loadPlaylist() {
  try {
    const data = await loadJSON("../album/tracklist.json");
    if (Array.isArray(data) && data.length > 0) {
      postMessage(`Loaded ${data.length} track playlist`, "success");
      return data;
    }
    postMessage("tracklist.json is empty; load audio manually", "warn");
  } catch (error) {
    postMessage("Playlist not found yet — use Load Audio or Test Tone", "warn");
    console.info("[app] playlist unavailable (expected before Phase 4)", error);
  }
  return [];
}

function resolveTrackSource(track) {
  if (!track?.src) return null;
  try {
    return new URL(`../album/${track.src}`, import.meta.url).toString();
  } catch (error) {
    console.warn("[app] failed to resolve track source", error);
    return null;
  }
}

function updateNowPlaying(track) {
  if (!track) {
    updateElementText(elements.nowPlayingTitle, "No track loaded");
    updateElementText(elements.statusMessage, "Drop an MP3 below or use test tone.");
    if (elements.art) {
      elements.art.src = "";
      elements.art.style.visibility = "hidden";
    }
    return;
  }
  updateElementText(elements.nowPlayingTitle, `${track.title || "Untitled"} — ${track.artist || "Unknown"}`);
  if (track.art && elements.art) {
    elements.art.src = new URL(`../album/${track.art}`, import.meta.url).toString();
    elements.art.style.visibility = "visible";
  } else if (elements.art) {
    elements.art.style.visibility = "hidden";
  }
}

function updatePlayButtonIcon(audio) {
  if (!elements.playBtn) return;
  elements.playBtn.textContent = audio.paused ? "►" : "❚❚";
}

function initControls(audio) {
  elements.playBtn?.addEventListener("click", () => {
    if (audio.paused) {
      audio.play().catch((error) => {
        postMessage("Audio playback blocked; user gesture required", "error");
        console.warn("[audio] play failed", error);
      });
    } else {
      audio.pause();
    }
  });

  audio.addEventListener("play", () => updatePlayButtonIcon(audio));
  audio.addEventListener("pause", () => updatePlayButtonIcon(audio));
}

async function bootstrap() {
  postMessage("Bootstrapping…");

  const meta = await loadJSON("../models/meta.json");
  const sessionInfo = await createDecoderSession(meta, { preferInt8: true });

  const playlist = await loadPlaylist();
  const audioPipeline = createAudioPipeline(elements.audio);
  audioPipeline.ensureConnected();

  const featureExtractor = createFeatureExtractor({
    analyser: audioPipeline.analyser,
    smoothing: Number.parseFloat(elements.smoothing?.value || "0.9"),
  });

  const controller = createController({
    latentShape: meta.latent_shape,
    sensitivity: Number.parseFloat(elements.sensitivity?.value || "0.75"),
    smoothing: Number.parseFloat(elements.smoothing?.value || "0.9"),
  });

  let rendererMode = elements.renderer?.value || "canvas";
  let viz = await createCanvasViz({
    sessionInfo,
    meta,
    canvas: elements.stage2d,
    size: Number.parseInt(elements.quality?.value || "384", 10),
  });
  let recorder = createRecorder({
    element: viz.getCanvas(),
    onReady: (blob) => {
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `noise-to-signal-${Date.now()}.webm`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      postMessage("Recording saved to downloads", "success");
      elements.recordBtn.textContent = "● Record";
    },
  });

  let currentIndex = Number.parseInt(localStorage.getItem(STORAGE_KEYS.index) || "0", 10);
  let resumeTime = Number.parseFloat(localStorage.getItem(STORAGE_KEYS.time) || "0");

  function persistPlaybackState() {
    if (!Number.isFinite(elements.audio.duration) || Number.isNaN(elements.audio.duration)) {
      return;
    }
    localStorage.setItem(STORAGE_KEYS.index, String(currentIndex));
    localStorage.setItem(STORAGE_KEYS.time, elements.audio.currentTime.toFixed(2));
  }

  function loadTrack(index) {
    if (!playlist.length) return;
    currentIndex = (index + playlist.length) % playlist.length;
    const track = playlist[currentIndex];
    const src = resolveTrackSource(track);
    if (!src) {
      postMessage(`Failed to load track ${track.title || ""}`, "error");
      return;
    }
    updateNowPlaying(track);
    elements.audio.src = src;
    elements.audio.load();
    if (resumeTime > 0) {
      elements.audio.currentTime = resumeTime;
      resumeTime = 0;
    } else {
      elements.audio.currentTime = 0;
    }
    elements.audio.play().catch((error) => {
      console.warn("[audio] autoplay prevented", error);
    });
  }

  if (playlist.length) {
    loadTrack(currentIndex);
  } else {
    updateNowPlaying(null);
  }

  initControls(elements.audio);

  elements.prevBtn?.addEventListener("click", () => {
    if (!playlist.length) return;
    loadTrack(currentIndex - 1);
  });

  elements.nextBtn?.addEventListener("click", () => {
    if (!playlist.length) return;
    loadTrack(currentIndex + 1);
  });

  elements.audio.addEventListener("ended", () => {
    if (playlist.length) {
      loadTrack(currentIndex + 1);
    }
  });

  elements.audio.addEventListener("timeupdate", () => {
    if (elements.audio.duration) {
      const value = elements.audio.currentTime / elements.audio.duration;
      elements.seek.value = String(Math.floor(value * 1000));
      updateElementText(elements.time, `${formatTime(elements.audio.currentTime)} / ${formatTime(elements.audio.duration)}`);
      persistPlaybackState();
    }
  });

  elements.seek?.addEventListener("input", () => {
    if (elements.audio.duration) {
      const target = (Number.parseInt(elements.seek.value, 10) / 1000) * elements.audio.duration;
      elements.audio.currentTime = target;
      persistPlaybackState();
    }
  });

  elements.quality?.addEventListener("change", () => {
    const size = Number.parseInt(elements.quality.value, 10);
    viz.resize(size);
  });

  async function switchRenderer(nextMode) {
    if (rendererMode === nextMode) return;
    rendererMode = nextMode;
    viz.dispose();

    if (rendererMode === "three") {
      elements.stage2d.hidden = true;
      elements.threeRoot.hidden = false;
      viz = await createThreeViz({
        sessionInfo,
        meta,
        root: elements.threeRoot,
        size: Number.parseInt(elements.quality.value, 10),
      });
    } else {
      elements.stage2d.hidden = false;
      elements.threeRoot.hidden = true;
      viz = await createCanvasViz({
        sessionInfo,
        meta,
        canvas: elements.stage2d,
        size: Number.parseInt(elements.quality.value, 10),
      });
    }
    recorder.updateElement(viz.getCanvas());
  }

  elements.renderer?.addEventListener("change", async () => {
    postMessage(`Switching to ${elements.renderer.value} renderer…`);
    await switchRenderer(elements.renderer.value);
    postMessage(`Renderer ready: ${elements.renderer.value}`);
  });

  elements.sensitivity?.addEventListener("input", () => {
    controller.setSensitivity(Number.parseFloat(elements.sensitivity.value));
  });

  elements.smoothing?.addEventListener("input", () => {
    const value = Number.parseFloat(elements.smoothing.value);
    controller.setSmoothing(value);
    featureExtractor.setSmoothing(value);
  });

  elements.recordBtn?.addEventListener("click", () => {
    if (!window.MediaRecorder) {
      postMessage("MediaRecorder not supported in this browser", "error");
      return;
    }
    if (recorder.isRecording()) {
      recorder.stop();
      elements.recordBtn.textContent = "● Record";
    } else {
      recorder.start();
      elements.recordBtn.textContent = "■ Stop";
      postMessage("Recording… click stop to save", "info");
    }
  });

  bindFileInputToAudio(elements.fileInput, elements.audio, ({ file }) => {
    updateElementText(elements.nowPlayingTitle, file.name);
    if (elements.art) {
      elements.art.style.visibility = "hidden";
    }
    postMessage(`Loaded local file: ${file.name}`, "success");
  });

  elements.testToneBtn?.addEventListener("click", () => {
    try {
      const tone = createTestTone(audioPipeline.context, { duration: 12, type: "sawtooth" });
      tone.output.connect(audioPipeline.analyser);
      tone.start();
      postMessage("Playing internal test tone", "success");
      audioPipeline.resume().catch(() => {});
    } catch (error) {
      console.error("[audio] test tone failed", error);
      postMessage("Test tone unavailable", "error");
    }
  });

  postMessage(`Decoder ready (${sessionInfo.modelType.toUpperCase()})`);

  const fpsState = { timestamp: performance.now(), accumulator: 0, frames: 0 };
  function loop() {
    const { vector, metrics } = featureExtractor.nextFrame();
    const latent = controller.update(vector, metrics);
    viz.render(latent);
    const fps = calculateFps({ now: performance.now(), previous: fpsState, frames: 1 });
    if (fps && elements.fps) {
      elements.fps.textContent = `${fps.toFixed(1)} fps`;
    }
    requestAnimationFrame(loop);
  }
  loop();
}

bootstrap().catch((error) => {
  console.error("[app] bootstrap failed", error);
  postMessage(`Initialization failed: ${error.message}`, "error");
});
