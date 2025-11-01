export function createRecorder({ element, onReady } = {}) {
  let recorder = null;
  let chunks = [];

  function requireStream() {
    if (!element) {
      throw new Error("Recording element is not configured");
    }
    if (!element.captureStream) {
      throw new Error("captureStream is not supported in this browser");
    }
    return element.captureStream(30);
  }

  function start() {
    if (recorder && recorder.state === "recording") {
      return;
    }
    const stream = requireStream();
    const mimeType =
      MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ?
        "video/webm;codecs=vp9" :
        "video/webm;codecs=vp8";
    recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 6_000_000 });
    chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType });
      chunks = [];
      if (typeof onReady === "function") {
        onReady(blob);
      } else {
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `noise-to-signal-${Date.now()}.webm`;
        anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      }
    };
    recorder.start(250);
  }

  function stop() {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }

  function isRecording() {
    return recorder?.state === "recording";
  }

  function updateElement(nextElement) {
    element = nextElement;
  }

  return {
    start,
    stop,
    isRecording,
    updateElement,
  };
}
