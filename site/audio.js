export function createAudioPipeline(audioElement, options = {}) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is not supported in this environment");
  }

  const context = new AudioContextClass();
  const analyser = context.createAnalyser();
  analyser.fftSize = options.fftSize || 2048;
  analyser.smoothingTimeConstant = 0;
  analyser.minDecibels = -100;
  analyser.maxDecibels = -10;

  let source = null;
  const gain = context.createGain();
  gain.gain.value = options.monitorGain ?? 1.0;

  function ensureConnected() {
    if (!source) {
      source = context.createMediaElementSource(audioElement);
      source.connect(analyser);
      analyser.connect(gain);
      gain.connect(context.destination);
    }
  }

  async function resume() {
    ensureConnected();
    if (context.state === "suspended") {
      await context.resume();
    }
  }

  audioElement.addEventListener("play", () => {
    resume().catch((error) => {
      console.warn("[audio] resume failed", error);
    });
  });

  return {
    context,
    analyser,
    resume,
    ensureConnected,
    setMonitorGain(value) {
      gain.gain.value = value;
    },
    disconnect() {
      if (source) {
        source.disconnect();
        analyser.disconnect();
        gain.disconnect();
        source = null;
      }
    },
  };
}

export function bindFileInputToAudio(inputElement, audioElement, onLoad) {
  if (!inputElement) return;
  inputElement.addEventListener("change", () => {
    const [file] = inputElement.files || [];
    if (!file) return;
    const url = URL.createObjectURL(file);
    audioElement.src = url;
    audioElement.play().catch((error) => {
      console.warn("[audio] autoplay blocked", error);
    });
    if (typeof onLoad === "function") {
      onLoad({ file, url });
    }
  });
}

export function createTestTone(context, { duration = 8, type = "sine" } = {}) {
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  oscillator.type = type;
  oscillator.frequency.value = 220;

  gain.gain.setValueAtTime(0.0001, context.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.35, context.currentTime + 0.1);
  gain.gain.exponentialRampToValueAtTime(0.05, context.currentTime + duration - 0.1);
  gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + duration);

  oscillator.connect(gain);
  gain.connect(context.destination);

  return {
    start() {
      oscillator.start();
      oscillator.stop(context.currentTime + duration);
    },
    stop() {
      oscillator.stop();
    },
    output: gain,
  };
}
