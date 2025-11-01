import { clamp, createSeededRandom, gaussianRandom, softmax } from "./utils.js";

const DEFAULT_ANCHORS = 6;

export function createController({
  latentShape,
  sensitivity = 0.75,
  smoothing = 0.9,
  wander = 0.25,
  anchorCount = DEFAULT_ANCHORS,
} = {}) {
  if (!latentShape || latentShape.length !== 3) {
    throw new Error("latentShape must be [channels, height, width]");
  }

  const latentSize = latentShape[0] * latentShape[1] * latentShape[2];
  const state = {
    latent: new Float32Array(latentSize),
    target: new Float32Array(latentSize),
    anchors: [],
    weights: new Float32Array(anchorCount),
    featureProjection: null,
    featureDim: 0,
    sensitivity: sensitivity,
    smoothing: clamp(smoothing, 0, 0.99),
    wander: clamp(wander, 0, 1),
    wanderPhase: 0,
    wanderSpeed: 0.35,
  };

  const rand = createSeededRandom(20251031);
  const gaussian = gaussianRandom(rand);

  for (let a = 0; a < anchorCount; a += 1) {
    const anchor = new Float32Array(latentSize);
    for (let i = 0; i < latentSize; i += 1) {
      anchor[i] = gaussian() * 0.8;
    }
    state.anchors.push(anchor);
  }

  function ensureProjection(featureDim) {
    if (state.featureProjection && state.featureDim === featureDim) {
      return;
    }
    state.featureDim = featureDim;
    state.featureProjection = new Float32Array(anchorCount * featureDim);
    for (let anchorIdx = 0; anchorIdx < anchorCount; anchorIdx += 1) {
      for (let f = 0; f < featureDim; f += 1) {
        const index = anchorIdx * featureDim + f;
        state.featureProjection[index] = gaussian() * 0.8;
      }
    }
  }

  function update(featureVector, metrics = {}) {
    const dim = featureVector.length;
    ensureProjection(dim);

    for (let anchorIdx = 0; anchorIdx < anchorCount; anchorIdx += 1) {
      let score = 0;
      for (let f = 0; f < dim; f += 1) {
        score += featureVector[f] * state.featureProjection[anchorIdx * dim + f];
      }
      state.weights[anchorIdx] = score * state.sensitivity;
    }

    const weights = softmax(Array.from(state.weights), 1.0);

    state.target.fill(0);
    for (let anchorIdx = 0; anchorIdx < anchorCount; anchorIdx += 1) {
      const weight = weights[anchorIdx] ?? 0;
      const anchor = state.anchors[anchorIdx];
      for (let i = 0; i < latentSize; i += 1) {
        state.target[i] += anchor[i] * weight;
      }
    }

    const rms = metrics.rms ?? 0;
    const centroid = metrics.centroidNorm ?? 0.5;
    state.wanderPhase += state.wanderSpeed * (0.5 + centroid * 0.5);
    const wanderStrength = state.wander * (0.2 + rms * 1.5);

    for (let i = 0; i < latentSize; i += 1) {
      const phase = state.wanderPhase + (i % latentShape[2]) * 0.02;
      state.target[i] += wanderStrength * Math.sin(phase);
    }

    const blend = 1 - clamp(state.smoothing, 0, 0.99);
    for (let i = 0; i < latentSize; i += 1) {
      const delta = state.target[i] - state.latent[i];
      state.latent[i] += delta * blend;
    }

    return state.latent;
  }

  return {
    update,
    setSensitivity(value) {
      state.sensitivity = clamp(value, 0.1, 4);
    },
    setSmoothing(value) {
      state.smoothing = clamp(value, 0, 0.99);
    },
    setWander(value) {
      state.wander = clamp(value, 0, 1);
    },
    getAnchorCount() {
      return anchorCount;
    },
  };
}
