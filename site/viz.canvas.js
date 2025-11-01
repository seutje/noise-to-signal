import { clamp, toRgbaBuffer } from "./utils.js";

export async function createCanvasViz({ sessionInfo, meta, canvas = document.getElementById("stage2d"), size = 384 }) {
  if (!sessionInfo || !sessionInfo.session) {
    throw new Error("Missing decoder session");
  }
  if (!canvas) {
    throw new Error("Canvas element not found");
  }

  const { session, inputName, outputName } = sessionInfo;
  const latentShape = meta.latent_shape;
  const latentSize = latentShape.reduce((acc, value) => acc * value, 1);
  const baseResolution = meta.image_resolution || 256;
  const normalization = meta.normalization || { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };

  canvas.width = baseResolution;
  canvas.height = baseResolution;

  const ctx = canvas.getContext("2d", { willReadFrequently: false });
  const offscreen = document.createElement("canvas");
  offscreen.width = baseResolution;
  offscreen.height = baseResolution;
  const offCtx = offscreen.getContext("2d");
  const imageData = offCtx.createImageData(baseResolution, baseResolution);

  let pending = false;
  let queued = null;
  let disposed = false;

  const latentBuffer = new Float32Array(latentSize);

  function setSize(nextSize) {
    const px = clamp(Number.parseInt(nextSize, 10) || baseResolution, 128, 1024);
    canvas.style.width = `${px}px`;
    canvas.style.height = `${px}px`;
  }

  setSize(size);

  async function decode(latentArray) {
    latentBuffer.set(latentArray);
    const feeds = {
      [inputName]: new window.ort.Tensor("float32", latentBuffer, [1, ...latentShape]),
    };
    const outputMap = await session.run(feeds);
    const tensor = outputMap[outputName];
    const data = tensor.data;
    const aligned = data instanceof Float32Array ? data : Float32Array.from(data);
    return aligned;
  }

  function draw(pixels) {
    const rgba = toRgbaBuffer(pixels, normalization);
    imageData.data.set(rgba);
    offCtx.putImageData(imageData, 0, 0);
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }

  async function process(latentArray) {
    if (disposed) return;
    pending = true;
    try {
      const pixels = await decode(latentArray);
      draw(pixels);
    } catch (error) {
      console.error("[viz.canvas] decode failed", error);
    } finally {
      pending = false;
      if (queued) {
        const next = queued;
        queued = null;
        process(next);
      }
    }
  }

  function render(latentArray) {
    if (disposed) return;
    const copy = new Float32Array(latentArray);
    if (pending) {
      queued = copy;
      return;
    }
    process(copy);
  }

  return {
    render,
    resize: setSize,
    dispose() {
      disposed = true;
      queued = null;
    },
    getCanvas() {
      return canvas;
    },
  };
}
