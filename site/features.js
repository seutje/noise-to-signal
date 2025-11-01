import { clamp, mappedRange, takeAverage } from "./utils.js";

function createArray(length, fillValue = 0) {
  return new Float32Array(length).fill(fillValue);
}

function computeRms(buffer) {
  let sumSq = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    const v = buffer[i];
    sumSq += v * v;
  }
  return Math.sqrt(sumSq / buffer.length);
}

function computeZeroCrossings(buffer) {
  let crossings = 0;
  for (let i = 1; i < buffer.length; i += 1) {
    if ((buffer[i - 1] >= 0 && buffer[i] < 0) || (buffer[i - 1] < 0 && buffer[i] >= 0)) {
      crossings += 1;
    }
  }
  return crossings / buffer.length;
}

function computeSpectralCentroid(magnitudes, sampleRate) {
  let num = 0;
  let den = 0;
  const factor = sampleRate / (2 * magnitudes.length);
  for (let i = 0; i < magnitudes.length; i += 1) {
    const mag = magnitudes[i];
    num += mag * i * factor;
    den += mag;
  }
  return den > 0 ? num / den : 0;
}

function computeSpectralFlatness(magnitudes) {
  let geo = 0;
  let arith = 0;
  const eps = 1e-9;
  for (let i = 0; i < magnitudes.length; i += 1) {
    const mag = Math.max(magnitudes[i], eps);
    geo += Math.log(mag);
    arith += mag;
  }
  const geometricMean = Math.exp(geo / magnitudes.length);
  const arithmeticMean = arith / magnitudes.length;
  return clamp(geometricMean / Math.max(arithmeticMean, eps), 0, 1);
}

function computeSpectralRolloff(magnitudes, sampleRate, percentile = 0.85) {
  const totalEnergy = magnitudes.reduce((acc, value) => acc + value, 0);
  const target = totalEnergy * percentile;
  let cumulative = 0;
  for (let i = 0; i < magnitudes.length; i += 1) {
    cumulative += magnitudes[i];
    if (cumulative >= target) {
      return (i / magnitudes.length) * (sampleRate / 2);
    }
  }
  return sampleRate / 2;
}

function computeBandEnergies(magnitudes, bands) {
  const energies = [];
  const binCount = magnitudes.length;
  for (let i = 0; i < bands.length - 1; i += 1) {
    const start = Math.floor(bands[i] * binCount);
    const end = Math.floor(bands[i + 1] * binCount);
    let sum = 0;
    for (let j = start; j < end; j += 1) {
      sum += magnitudes[j];
    }
    energies.push(sum / Math.max(end - start, 1));
  }
  return energies;
}

function computeMfcc(magnitudes, melFilters, dctMatrix) {
  const melEnergies = melFilters.map((filter) => {
    let sum = 0;
    for (let i = 0; i < magnitudes.length; i += 1) {
      sum += magnitudes[i] * filter[i];
    }
    return Math.log(Math.max(sum, 1e-9));
  });

  const coeffs = [];
  for (let i = 0; i < dctMatrix.length; i += 1) {
    let value = 0;
    const row = dctMatrix[i];
    for (let j = 0; j < melEnergies.length; j += 1) {
      value += row[j] * melEnergies[j];
    }
    coeffs.push(value);
  }
  return coeffs;
}

function createMelFilterBank(binCount, sampleRate, melBandCount) {
  const toMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  const fromMel = (mel) => 700 * (10 ** (mel / 2595) - 1);

  const nyquist = sampleRate / 2;
  const melMin = toMel(20);
  const melMax = toMel(nyquist);
  const melPoints = [];
  const filters = [];

  for (let i = 0; i < melBandCount + 2; i += 1) {
    melPoints.push(melMin + ((melMax - melMin) * i) / (melBandCount + 1));
  }

  const binFrequencies = melPoints.map((mel) => Math.floor((binCount + 1) * fromMel(mel) / sampleRate));

  for (let i = 1; i <= melBandCount; i += 1) {
    const filter = new Float32Array(binCount);
    for (let j = binFrequencies[i - 1]; j < binFrequencies[i]; j += 1) {
      filter[j] = (j - binFrequencies[i - 1]) / Math.max(binFrequencies[i] - binFrequencies[i - 1], 1);
    }
    for (let j = binFrequencies[i]; j < binFrequencies[i + 1]; j += 1) {
      filter[j] = (binFrequencies[i + 1] - j) / Math.max(binFrequencies[i + 1] - binFrequencies[i], 1);
    }
    filters.push(filter);
  }

  return filters;
}

function createDctMatrix(rows, cols) {
  const matrix = [];
  const factor = Math.PI / cols;
  for (let i = 0; i < rows; i += 1) {
    const row = new Float32Array(cols);
    for (let j = 0; j < cols; j += 1) {
      row[j] = Math.cos((i + 0.5) * j * factor);
    }
    matrix.push(row);
  }
  return matrix;
}

export function createFeatureExtractor({ analyser, smoothing = 0.9, bands = 8, mfccCount = 8 }) {
  const fftSize = analyser.fftSize;
  const binCount = fftSize / 2;
  const sampleRate = analyser.context?.sampleRate || 44100;

  const timeDomain = new Float32Array(fftSize);
  const frequencyDomain = new Float32Array(binCount);
  const magnitudes = new Float32Array(binCount);
  const smoothedVector = createArray(32);
  const melFilters = createMelFilterBank(binCount, sampleRate, mfccCount);
  const dctMatrix = createDctMatrix(mfccCount, mfccCount);
  const bandBoundaries = Array.from({ length: bands + 1 }, (_, i) => i / bands);

  let smoothingFactor = clamp(smoothing, 0, 0.99);

  const metrics = {
    rms: 0,
    centroidHz: 0,
    centroidNorm: 0,
    flatness: 0,
    rolloffHz: 0,
    zeroCrossings: 0,
    bandAverages: new Float32Array(bands),
  };

  function setSmoothing(value) {
    smoothingFactor = clamp(value, 0, 0.99);
  }

  function nextFrame() {
    analyser.getFloatTimeDomainData(timeDomain);
    analyser.getFloatFrequencyData(frequencyDomain);

    for (let i = 0; i < binCount; i += 1) {
      const db = frequencyDomain[i];
      magnitudes[i] = 10 ** (db / 20);
    }

    metrics.rms = computeRms(timeDomain);
    metrics.zeroCrossings = computeZeroCrossings(timeDomain);
    metrics.centroidHz = computeSpectralCentroid(magnitudes, sampleRate);
    metrics.centroidNorm = mappedRange(metrics.centroidHz, 50, sampleRate / 2, 0, 1);
    metrics.flatness = computeSpectralFlatness(magnitudes);
    metrics.rolloffHz = computeSpectralRolloff(magnitudes, sampleRate, 0.85);

    const bandEnergies = computeBandEnergies(magnitudes, bandBoundaries);
    metrics.bandAverages.set(bandEnergies);

    const mfcc = computeMfcc(magnitudes, melFilters, dctMatrix);

    const vector = [
      metrics.rms,
      metrics.zeroCrossings,
      metrics.centroidNorm,
      metrics.flatness,
      mappedRange(metrics.rolloffHz, 200, sampleRate / 2, 0, 1),
      ...bandEnergies,
      ...mfcc,
    ];

    while (vector.length < smoothedVector.length) {
      vector.push(0);
    }
    if (vector.length > smoothedVector.length) {
      vector.length = smoothedVector.length;
    }

    const alpha = 1 - smoothingFactor;
    for (let i = 0; i < smoothedVector.length; i += 1) {
      const value = vector[i] ?? 0;
      smoothedVector[i] = smoothedVector[i] * smoothingFactor + value * alpha;
    }

    return {
      vector: smoothedVector,
      metrics,
    };
  }

  return {
    nextFrame,
    setSmoothing,
    getAverages: () => ({
      rms: metrics.rms,
      centroid: metrics.centroidHz,
      flatness: metrics.flatness,
    }),
    debugVector: smoothedVector,
  };
}
