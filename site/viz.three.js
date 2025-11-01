import { clamp } from "./utils.js";

export async function createThreeViz({
  sessionInfo,
  meta,
  root = document.getElementById("three-root"),
  size = 384,
}) {
  const THREE = window.THREE;
  if (!THREE) {
    throw new Error("Three.js is required for the WebGL renderer");
  }
  if (!root) {
    throw new Error("Three.js root element missing");
  }
  if (!sessionInfo || !sessionInfo.session) {
    throw new Error("Missing decoder session");
  }

  const { session, inputName, outputName } = sessionInfo;
  const latentShape = meta.latent_shape;
  const latentSize = latentShape.reduce((acc, value) => acc * value, 1);
  const baseResolution = meta.image_resolution || 256;
  const normalization = meta.normalization || { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  root.replaceChildren(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const geometry = new THREE.PlaneGeometry(2, 2);

  const pixelCount = baseResolution * baseResolution;
  const rgbData = new Uint8Array(pixelCount * 3);
  const smoothedData = new Float32Array(pixelCount * 3);
  const latentBuffer = new Float32Array(latentSize);

  const texture = new THREE.DataTexture(rgbData, baseResolution, baseResolution, THREE.RGBFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;

  const quadMaterial = new THREE.ShaderMaterial({
    uniforms: {
      tDiffuse: { value: texture },
      uTime: { value: 0 },
      uIntensity: { value: 0.18 },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      precision mediump float;
      varying vec2 vUv;
      uniform sampler2D tDiffuse;
      uniform float uTime;
      uniform float uIntensity;

      vec3 bloom(vec3 color) {
        float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
        vec3 glow = smoothstep(0.65, 1.0, color) * lum * uIntensity;
        return color + glow;
      }

      void main() {
        vec2 ripple = vec2(
          sin(vUv.y * 10.0 + uTime * 0.8) * 0.0025,
          cos(vUv.x * 10.0 + uTime * 0.6) * 0.0025
        );
        vec3 base = texture2D(tDiffuse, vUv + ripple).rgb;
        base = pow(base, vec3(0.92));
        vec3 color = bloom(base);
        gl_FragColor = vec4(color, 1.0);
      }
    `,
  });

  const quad = new THREE.Mesh(geometry, quadMaterial);
  scene.add(quad);

  let disposed = false;
  let pending = false;
  let queued = null;
  let time = 0;

  function setSize(nextSize) {
    const px = clamp(Number.parseInt(nextSize, 10) || baseResolution, 128, 1024);
    renderer.setSize(px, px, false);
    renderer.domElement.style.width = `${px}px`;
    renderer.domElement.style.height = `${px}px`;
  }

  setSize(size);
  root.hidden = false;

  async function decode(latentArray) {
    latentBuffer.set(latentArray);
    const feeds = {
      [inputName]: new window.ort.Tensor("float32", latentBuffer, [1, ...latentShape]),
    };
    const outputMap = await session.run(feeds);
    const tensor = outputMap[outputName];
    return tensor.data instanceof Float32Array ? tensor.data : Float32Array.from(tensor.data);
  }

  function applyAfterimage(pixels) {
    const { mean, std } = normalization;
    const decay = 0.82;
    const bleed = 0.15;

    for (let i = 0; i < pixelCount; i += 1) {
      const r = clamp(Math.round((pixels[i * 3 + 0] * std[0] + mean[0]) * 255), 0, 255);
      const g = clamp(Math.round((pixels[i * 3 + 1] * std[1] + mean[1]) * 255), 0, 255);
      const b = clamp(Math.round((pixels[i * 3 + 2] * std[2] + mean[2]) * 255), 0, 255);

      const sr = smoothedData[i * 3 + 0] = smoothedData[i * 3 + 0] * decay + r * (1 - decay);
      const sg = smoothedData[i * 3 + 1] = smoothedData[i * 3 + 1] * decay + g * (1 - decay);
      const sb = smoothedData[i * 3 + 2] = smoothedData[i * 3 + 2] * decay + b * (1 - decay);

      rgbData[i * 3 + 0] = clamp(Math.round(r * (1 - bleed) + sr * bleed), 0, 255);
      rgbData[i * 3 + 1] = clamp(Math.round(g * (1 - bleed) + sg * bleed), 0, 255);
      rgbData[i * 3 + 2] = clamp(Math.round(b * (1 - bleed) + sb * bleed), 0, 255);
    }
    texture.needsUpdate = true;
  }

  function renderFrame() {
    if (disposed) return;
    quadMaterial.uniforms.uTime.value = time;
    renderer.render(scene, camera);
    time += 1 / 60;
  }

  async function process(latentArray) {
    if (disposed) return;
    pending = true;
    try {
      const pixels = await decode(latentArray);
      applyAfterimage(pixels);
      renderFrame();
    } catch (error) {
      console.error("[viz.three] decode failed", error);
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
      texture.dispose();
      quadMaterial.dispose();
      geometry.dispose();
      renderer.dispose();
      scene.clear();
      root.hidden = true;
    },
    getCanvas() {
      return renderer.domElement;
    },
  };
}
