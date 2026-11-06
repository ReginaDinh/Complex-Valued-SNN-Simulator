// Complex number utilities for Float32Array and tuple [Re, Im]
export const complex = {
  zero: (): [number, number] => [0, 0],
  fromPolar: (r: number, theta: number): [number, number] => [r * Math.cos(theta), r * Math.sin(theta)],
  create: (re: number, im: number): [number, number] => [re, im],
  add: (c1: [number, number], c2: [number, number]): [number, number] => [c1[0] + c2[0], c1[1] + c2[1]],
  sub: (c1: [number, number], c2: [number, number]): [number, number] => [c1[0] - c2[0], c1[1] - c2[1]],
  mul: (c1: [number, number], c2: [number, number]): [number, number] => [
    c1[0] * c2[0] - c1[1] * c2[1],
    c1[0] * c2[1] + c1[1] * c2[0],
  ],
  // Karatsuba complex multiplication using 3 real multiplications instead of 4
  mulKaratsuba: (c1: [number, number], c2: [number, number]): [number, number] => {
    const k1 = c1[0] * c2[0];
    const k2 = c1[1] * c2[1];
    const k3 = (c1[0] + c1[1]) * (c2[0] + c2[1]);
    return [k1 - k2, k3 - k1 - k2];
  },
  mag: (re: number, im: number): number => Math.sqrt(re * re + im * im),
  angle: (re: number, im: number): number => Math.atan2(im, re),
};

export type Complex = [number, number];
export type SurrogateType = 'von_mises' | 'fast_sigmoid' | 'atan';

// Changing V_th now rescales the surrogate width proportionally; results for threshold = 1.0 are unchanged.
export function computeSurrogateDerivative(
  uMag: number,
  threshold: number,
  type: SurrogateType = 'von_mises',
  sharpness: number = 2.5
): number {
  // Relative distance to the circular threshold — identical to the PyTorch VonMisesSurrogate
  const diff = (uMag - threshold) / Math.max(threshold, 1e-6);
  if (type === 'fast_sigmoid') {
    const denom = 1.0 + sharpness * Math.abs(diff);
    return 1.0 / (denom * denom);
  }
  if (type === 'atan') {
    const arg = sharpness * Math.PI * diff * 0.5;
    return 1.0 / (1.0 + arg * arg);
  }
  // Von Mises directional surrogate centered at |U| = threshold
  // Normalized by modified Bessel I0 approximation so peak gradient is well-scaled
  const clampedDiff = Math.max(-1.0, Math.min(1.0, diff));
  const i0Approx = Math.exp(sharpness) / Math.sqrt(2 * Math.PI * Math.max(0.5, sharpness));
  return Math.exp(sharpness * Math.cos(Math.PI * clampedDiff)) / (2 * Math.PI * i0Approx);
}

export interface SimulationPerfSnapshot {
  lastComputeMs: number;
  avgComputeMs: number;
  peakComputeMs: number;
  totalCalls: number;
  totalMacs: number;
  totalSpikes: number;
  activeTensorBytes: number;
  lastOpLabel: string;
  lastTimestamp: number;
  recentComputeHistory: number[];
}

class SimulationPerformanceTracker {
  lastComputeMs: number = 0;
  avgComputeMs: number = 0;
  peakComputeMs: number = 0;
  totalCalls: number = 0;
  totalMacs: number = 0;
  totalSpikes: number = 0;
  activeTensorBytes: number = 0;
  lastOpLabel: string = '—';
  lastTimestamp: number = performance.now();
  recentComputeHistory: number[] = [];

  private batchStartMs: number = 0;
  private batchAccumMs: number = 0;
  private batchMacs: number = 0;
  private batchSpikes: number = 0;
  private inHighLevelOp: boolean = false;

  beginOp() {
    this.inHighLevelOp = true;
    this.batchStartMs = performance.now();
    this.batchMacs = 0;
    this.batchSpikes = 0;
  }

  endOp(label: string, tensorBytes: number) {
    const elapsed = Math.max(0.01, performance.now() - this.batchStartMs);
    this.inHighLevelOp = false;
    this.recordSample(elapsed, this.batchMacs, this.batchSpikes, tensorBytes, label);
  }

  recordMicroStep(elapsedMs: number, macs: number, spiked: boolean) {
    if (this.inHighLevelOp) {
      this.batchMacs += macs;
      if (spiked) this.batchSpikes += 1;
      return;
    }
    this.batchAccumMs += elapsedMs;
    this.batchMacs += macs;
    if (spiked) this.batchSpikes += 1;

    const now = performance.now();
    if (now - this.lastTimestamp > 40 || this.batchAccumMs > 0.08) {
      this.recordSample(
        Math.max(0.02, this.batchAccumMs),
        this.batchMacs,
        this.batchSpikes,
        2048,
        'cLIF.step (Stream)'
      );
      this.batchAccumMs = 0;
      this.batchMacs = 0;
      this.batchSpikes = 0;
    }
  }

  recordSample(
    computeMs: number,
    macs: number,
    spikes: number,
    tensorBytes: number,
    label: string
  ) {
    this.lastComputeMs = computeMs;
    this.avgComputeMs =
      this.totalCalls === 0 ? computeMs : this.avgComputeMs * 0.82 + computeMs * 0.18;
    this.peakComputeMs = Math.max(this.peakComputeMs * 0.995, computeMs);
    this.totalCalls += 1;
    this.totalMacs += macs;
    this.totalSpikes += spikes;
    this.activeTensorBytes = tensorBytes;
    this.lastOpLabel = label;
    this.lastTimestamp = performance.now();
    this.recentComputeHistory = [...this.recentComputeHistory.slice(-19), computeMs];
  }

  resetPeak() {
    this.lastComputeMs = 0;
    this.avgComputeMs = 0;
    this.peakComputeMs = 0;
    this.totalCalls = 0;
    this.totalMacs = 0;
    this.totalSpikes = 0;
    this.recentComputeHistory = [];
    this.batchAccumMs = 0;
    this.batchMacs = 0;
    this.batchSpikes = 0;
  }

  getSnapshot(): SimulationPerfSnapshot {
    return {
      lastComputeMs: this.lastComputeMs,
      avgComputeMs: this.avgComputeMs,
      peakComputeMs: this.peakComputeMs,
      totalCalls: this.totalCalls,
      totalMacs: this.totalMacs,
      totalSpikes: this.totalSpikes,
      activeTensorBytes: this.activeTensorBytes,
      lastOpLabel: this.lastOpLabel,
      lastTimestamp: this.lastTimestamp,
      recentComputeHistory: this.recentComputeHistory,
    };
  }
}

export const perfTracker = new SimulationPerformanceTracker();

// Complex-Valued Leaky Integrate-and-Fire Neuron (cLIF)
export class ComplexLIFNeuron {
  uRe: number = 0;
  uIm: number = 0;
  preResetRe: number = 0;
  preResetIm: number = 0;

  leakMag: number;
  leakPhase: number;
  threshold: number;
  resetMode: 'soft' | 'hard';

  leakRe: number;
  leakIm: number;

  constructor(
    leakMag: number = 0.9,
    leakPhase: number = 0.1,
    threshold: number = 1.0,
    resetMode: 'soft' | 'hard' = 'soft'
  ) {
    this.leakMag = Math.max(0, Math.min(0.999, leakMag));
    this.leakPhase = leakPhase;
    this.threshold = Math.max(0.05, threshold);
    this.resetMode = resetMode;
    this.leakRe = this.leakMag * Math.cos(leakPhase);
    this.leakIm = this.leakMag * Math.sin(leakPhase);
  }

  // Update parameters in place without touching membrane state
  setParams(
    leakMag: number,
    leakPhase: number,
    threshold: number,
    resetMode: 'soft' | 'hard'
  ) {
    this.leakMag = Math.max(0, Math.min(0.999, leakMag));
    this.leakPhase = leakPhase;
    this.threshold = Math.max(0.05, threshold);
    this.resetMode = resetMode;
    this.leakRe = this.leakMag * Math.cos(leakPhase);
    this.leakIm = this.leakMag * Math.sin(leakPhase);
  }

  reset() {
    this.uRe = 0;
    this.uIm = 0;
    this.preResetRe = 0;
    this.preResetIm = 0;
  }

  step(
    inputRe: number,
    inputIm: number,
    track: boolean = true
  ): {
    spiked: boolean;
    spikeValRe: number;
    spikeValIm: number;
    preResetMag: number;
    preResetAngle: number;
  } {
    const t0 = track ? performance.now() : 0;
    // u_pre(t) = beta * u(t-1) + input(t), where beta = leakMag * e^{j * leakPhase}
    const nextRe = this.leakRe * this.uRe - this.leakIm * this.uIm + inputRe;
    const nextIm = this.leakRe * this.uIm + this.leakIm * this.uRe + inputIm;

    this.preResetRe = nextRe;
    this.preResetIm = nextIm;
    this.uRe = nextRe;
    this.uIm = nextIm;

    const mag = Math.sqrt(nextRe * nextRe + nextIm * nextIm);
    const angle = Math.atan2(nextIm, nextRe);

    if (mag >= this.threshold) {
      const spikeValRe = Math.cos(angle);
      const spikeValIm = Math.sin(angle);

      if (this.resetMode === 'hard') {
        this.uRe = 0;
        this.uIm = 0;
      } else {
        // Soft reset: subtract V_th in the exact phase direction e^{j * angle}
        this.uRe -= this.threshold * spikeValRe;
        this.uIm -= this.threshold * spikeValIm;
      }
      if (track) perfTracker.recordMicroStep(performance.now() - t0, 4, true);
      return {
        spiked: true,
        spikeValRe,
        spikeValIm,
        preResetMag: mag,
        preResetAngle: angle,
      };
    }

    if (track) perfTracker.recordMicroStep(performance.now() - t0, 4, false);
    return {
      spiked: false,
      spikeValRe: 0,
      spikeValIm: 0,
      preResetMag: mag,
      preResetAngle: angle,
    };
  }
}

export interface SignalSample {
  inputSpikeSequence: Float32Array[]; // array of [inFeatures * 2]
  target: number; // 0 or 1
  carrierFreq: number;
  spatialPhaseShift: number;
}

/**
 * Generates multi-channel complex spectral signal samples without Nyquist aliasing.
 * Normalized time t_norm = t / timeSteps in [0, 1) so frequencies up to timeSteps/2 cycles per window
 * are properly resolved without integer-step aliasing.
 * - Class 0: Coherent carrier at freq0 cycles per window (zero inter-channel phase gradient)
 * - Class 1: Carrier at freq1 cycles per window with inter-channel spatial phase gradient `phaseShift`
 */
export function generateSpectralSignal(
  targetClass: number,
  timeSteps: number,
  inFeatures: number,
  freq0: number = 1.0,
  freq1: number = 4.5,
  phaseShift: number = Math.PI / 2,
  noiseLevel: number = 0.15,
  rng: () => number = Math.random
): SignalSample {
  const sequence: Float32Array[] = [];
  const baseFreq = targetClass === 0 ? freq0 : freq1;
  const activePhaseShift = targetClass === 0 ? 0.0 : phaseShift;

  for (let t = 0; t < timeSteps; t++) {
    const tNorm = t / timeSteps; // Normalized time in [0, 1) -> avoids 2*pi*integer aliasing
    const stepArr = new Float32Array(inFeatures * 2);

    for (let i = 0; i < inFeatures; i++) {
      const isSignalChannel = i < Math.floor((inFeatures * 3) / 4); // First 12 channels carry structured signal

      if (isSignalChannel) {
        const cleanMag = 0.85 + (rng() - 0.5) * 0.15;
        const channelPhaseOffset = i * activePhaseShift;
        const phase = 2 * Math.PI * baseFreq * tNorm + channelPhaseOffset;

        const noiseRe = (rng() - 0.5) * 2 * noiseLevel;
        const noiseIm = (rng() - 0.5) * 2 * noiseLevel;

        stepArr[i * 2] = cleanMag * Math.cos(phase) + noiseRe;
        stepArr[i * 2 + 1] = cleanMag * Math.sin(phase) + noiseIm;
      } else {
        // Background clutter channels (uncorrelated random phase noise)
        const clutterMag = rng() * (0.35 + noiseLevel * 0.5);
        const clutterPhase = rng() * 2 * Math.PI;
        stepArr[i * 2] = clutterMag * Math.cos(clutterPhase);
        stepArr[i * 2 + 1] = clutterMag * Math.sin(clutterPhase);
      }
    }
    sequence.push(stepArr);
  }

  return {
    inputSpikeSequence: sequence,
    target: targetClass,
    carrierFreq: baseFreq,
    spatialPhaseShift: activePhaseShift,
  };
}

export interface ForwardResult {
  surrogateRates: number[];
  spikeCounts: number[];
  firingRates: number[];
  allPotentials: Float32Array[]; // Post-reset membrane potentials [t][o*2, o*2+1]
  preResetPotentials: Float32Array[]; // Pre-reset membrane potentials [t][o*2, o*2+1]
  spikeTrains: boolean[][]; // [t][o]
  phaseCoherence: number[]; // Phase locking value (PLV) per output neuron
  realMultiplications: number;
}

// Complex-Valued Spiking Neural Network with BPTT & Surrogate Gradients
export class TSComplexSNN {
  inFeatures: number;
  outFeatures: number;
  useKaratsuba: boolean = true;
  surrogateType: SurrogateType = 'von_mises';

  // Weights: [outFeatures][inFeatures * 2]
  weights: Float32Array[];
  neurons: ComplexLIFNeuron[];

  constructor(
    inFeatures: number,
    outFeatures: number,
    leakMag: number = 0.85,
    leakPhase: number = 0.12,
    threshold: number = 1.0,
    resetMode: 'soft' | 'hard' = 'soft',
    useKaratsuba: boolean = true,
    surrogateType: SurrogateType = 'von_mises'
  ) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.useKaratsuba = useKaratsuba;
    this.surrogateType = surrogateType;
    this.weights = [];

    for (let o = 0; o < outFeatures; o++) {
      const row = new Float32Array(inFeatures * 2);
      for (let i = 0; i < inFeatures; i++) {
        // Complex Xavier / Glorot initialization
        const mag = (0.3 + Math.random() * 0.7) * Math.sqrt(1.5 / inFeatures);
        const phase = Math.random() * 2 * Math.PI - Math.PI;
        row[i * 2] = mag * Math.cos(phase);
        row[i * 2 + 1] = mag * Math.sin(phase);
      }
      this.weights.push(row);
    }

    this.neurons = [];
    for (let o = 0; o < outFeatures; o++) {
      this.neurons.push(new ComplexLIFNeuron(leakMag, leakPhase, threshold, resetMode));
    }
  }

  reset() {
    this.neurons.forEach(n => n.reset());
  }

  forward(inputSequence: Float32Array[], record: boolean = true): ForwardResult {
    const t0 = performance.now();
    this.reset();
    const timeSteps = inputSequence.length;

    const spikeCounts = new Float32Array(this.outFeatures);
    const surrogateRates = new Float32Array(this.outFeatures);
    const phaseSumRe = new Float32Array(this.outFeatures);
    const phaseSumIm = new Float32Array(this.outFeatures);

    const allPotentials: Float32Array[] = [];
    const preResetPotentials: Float32Array[] = [];
    const spikeTrains: boolean[][] = [];

    let realMultiplications = 0;
    const multsPerSynapse = this.useKaratsuba ? 3 : 4;

    for (let t = 0; t < timeSteps; t++) {
      const input = inputSequence[t];
      const stepPotentials = new Float32Array(this.outFeatures * 2);
      const stepPreReset = new Float32Array(this.outFeatures * 2);
      const stepSpikes: boolean[] = [];

      for (let o = 0; o < this.outFeatures; o++) {
        let sumRe = 0;
        let sumIm = 0;

        for (let i = 0; i < this.inFeatures; i++) {
          const wRe = this.weights[o][i * 2];
          const wIm = this.weights[o][i * 2 + 1];
          const xRe = input[i * 2];
          const xIm = input[i * 2 + 1];

          if (this.useKaratsuba) {
            const k1 = wRe * xRe;
            const k2 = wIm * xIm;
            const k3 = (wRe + wIm) * (xRe + xIm);
            sumRe += k1 - k2;
            sumIm += k3 - k1 - k2;
          } else {
            sumRe += wRe * xRe - wIm * xIm;
            sumIm += wRe * xIm + wIm * xRe;
          }
          realMultiplications += multsPerSynapse;
        }

        const stepRes = this.neurons[o].step(sumRe, sumIm, false);
        stepSpikes.push(stepRes.spiked);
        if (stepRes.spiked) {
          spikeCounts[o]++;
        }

        // Store pre-reset potential for accurate readout & BPTT surrogate gradient
        stepPreReset[o * 2] = this.neurons[o].preResetRe;
        stepPreReset[o * 2 + 1] = this.neurons[o].preResetIm;

        // Store post-reset potential for membrane trajectory visualization
        stepPotentials[o * 2] = this.neurons[o].uRe;
        stepPotentials[o * 2 + 1] = this.neurons[o].uIm;

        // Readout combines pre-reset membrane amplitude (normalized by threshold) + spike emission bonus
        const normalizedMag = stepRes.preResetMag / this.neurons[o].threshold;
        surrogateRates[o] += normalizedMag + (stepRes.spiked ? 0.5 : 0);

        phaseSumRe[o] += Math.cos(stepRes.preResetAngle);
        phaseSumIm[o] += Math.sin(stepRes.preResetAngle);
      }

      allPotentials.push(stepPotentials);
      preResetPotentials.push(stepPreReset);
      spikeTrains.push(stepSpikes);
    }

    const firingRates: number[] = [];
    const phaseCoherence: number[] = [];

    for (let o = 0; o < this.outFeatures; o++) {
      surrogateRates[o] = surrogateRates[o] / timeSteps;
      firingRates.push(spikeCounts[o] / timeSteps);
      const plv =
        Math.sqrt(phaseSumRe[o] * phaseSumRe[o] + phaseSumIm[o] * phaseSumIm[o]) /
        Math.max(1, timeSteps);
      phaseCoherence.push(plv);
    }

    if (record) {
      const totalSpikesInPass = spikeCounts.reduce((acc, v) => acc + v, 0);
      const tensorBytes =
        (this.outFeatures * this.inFeatures * 2 +
          timeSteps * (this.inFeatures * 2 + this.outFeatures * 4)) *
        4;
      perfTracker.recordSample(
        Math.max(0.02, performance.now() - t0),
        realMultiplications,
        totalSpikesInPass,
        tensorBytes,
        this.useKaratsuba ? 'SNN.forward (Karatsuba 3M)' : 'SNN.forward (Standard 4M)'
      );
    }

    return {
      surrogateRates: Array.from(surrogateRates),
      spikeCounts: Array.from(spikeCounts),
      firingRates,
      allPotentials,
      preResetPotentials,
      spikeTrains,
      phaseCoherence,
      realMultiplications,
    };
  }

  evaluateLoss(sample: SignalSample, record: boolean = true): {
    loss: number;
    predictions: number[];
    firingRates: number[];
    phaseCoherence: number[];
    forward: ForwardResult;
  } {
    const forward = this.forward(sample.inputSpikeSequence, record);
    const { surrogateRates, firingRates, phaseCoherence } = forward;

    // Temperature-scaled softmax for crisp class probability estimation
    const temp = 2.5;
    const scaled = surrogateRates.map(r => r * temp);
    const maxRate = Math.max(...scaled);
    const exps = scaled.map(r => Math.exp(r - maxRate));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / (sumExps || 1e-9));

    const loss = -Math.log(probs[sample.target] + 1e-15);
    return { loss, predictions: probs, firingRates, phaseCoherence, forward };
  }

  trainStep(sample: SignalSample, learningRate: number, record: boolean = true): number {
    const t0 = performance.now();
    const timeSteps = sample.inputSpikeSequence.length;
    const {
      surrogateRates,
      preResetPotentials,
      spikeTrains,
      spikeCounts,
      realMultiplications,
    } = this.forward(sample.inputSpikeSequence, false);

    const temp = 2.5;
    const scaled = surrogateRates.map(r => r * temp);
    const maxRate = Math.max(...scaled);
    const exps = scaled.map(r => Math.exp(r - maxRate));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / (sumExps || 1e-9));
    const loss = -Math.log(probs[sample.target] + 1e-15);

    const dL_ds = new Float32Array(this.outFeatures);
    for (let o = 0; o < this.outFeatures; o++) {
      dL_ds[o] = (probs[o] - (o === sample.target ? 1 : 0)) * temp;
    }

    const betaRe = this.neurons[0].leakRe;
    const betaIm = this.neurons[0].leakIm;
    const threshold = this.neurons[0].threshold;
    const isHardReset = this.neurons[0].resetMode === 'hard';

    // Compute complex eligibility trace per output neuron & input channel
    // accounting for complex leak rotation and post-spike reset attenuation
    for (let o = 0; o < this.outFeatures; o++) {
      for (let i = 0; i < this.inFeatures; i++) {
        let fRe = 0;
        let fIm = 0;
        let gradRe = 0;
        let gradIm = 0;

        for (let t = 0; t < timeSteps; t++) {
          const xRe = sample.inputSpikeSequence[t][i * 2];
          const xIm = sample.inputSpikeSequence[t][i * 2 + 1];

          // Eligibility trace update: E(t) = beta * E(t-1) + X(t)
          const nextFRe = betaRe * fRe - betaIm * fIm + xRe;
          const nextFIm = betaRe * fIm + betaIm * fRe + xIm;
          fRe = nextFRe;
          fIm = nextFIm;

          // Use pre-reset potential U_pre(t) for Wirtinger gradient & surrogate derivative
          const uRe = preResetPotentials[t][o * 2];
          const uIm = preResetPotentials[t][o * 2 + 1];
          const mag = Math.sqrt(uRe * uRe + uIm * uIm) || 1e-9;

          const surrDeriv = computeSurrogateDerivative(mag, threshold, this.surrogateType, 2.5);
          // ds/d|U| for readout s = |U|/V_th + 0.5 * H(|U| - V_th), with H' ≈ surrogate
          const effectiveScale = 1.0 / threshold + 0.5 * surrDeriv;

          // Wirtinger derivative d|U|/dW_re and d|U|/dW_im:
          // Since U = W * E => U_re = W_re E_re - W_im E_im, U_im = W_re E_im + W_im E_re
          // d|U|/dW_re = (U_re * E_re + U_im * E_im) / |U|
          // d|U|/dW_im = (-U_re * E_im + U_im * E_re) / |U|
          const dU_dWre = ((uRe * fRe + uIm * fIm) / mag) * effectiveScale;
          const dU_dWim = ((-uRe * fIm + uIm * fRe) / mag) * effectiveScale;

          gradRe += dU_dWre;
          gradIm += dU_dWim;

          // If neuron spiked at step t, attenuate eligibility trace according to reset mode
          if (spikeTrains[t][o]) {
            if (isHardReset) {
              fRe = 0;
              fIm = 0;
            } else {
              const retainRatio = Math.max(0, (mag - threshold) / mag);
              fRe *= retainRatio;
              fIm *= retainRatio;
            }
          }
        }

        gradRe = (gradRe / timeSteps) * dL_ds[o];
        gradIm = (gradIm / timeSteps) * dL_ds[o];

        // Complex norm gradient clipping for numerical stability
        const gradMag = Math.sqrt(gradRe * gradRe + gradIm * gradIm);
        const maxGrad = 3.0;
        if (gradMag > maxGrad) {
          const scale = maxGrad / gradMag;
          gradRe *= scale;
          gradIm *= scale;
        }

        this.weights[o][i * 2] -= learningRate * gradRe;
        this.weights[o][i * 2 + 1] -= learningRate * gradIm;
      }
    }

    if (record) {
      const totalSpikes = spikeCounts.reduce((acc, v) => acc + v, 0);
      const bpttMacs = realMultiplications + timeSteps * this.outFeatures * this.inFeatures * 8;
      const bpttTensorBytes =
        (this.outFeatures * this.inFeatures * 6 +
          timeSteps * (this.inFeatures * 2 + this.outFeatures * 6)) *
        4;
      perfTracker.recordSample(
        Math.max(0.05, performance.now() - t0),
        bpttMacs,
        totalSpikes,
        bpttTensorBytes,
        `BPTT.trainStep (${this.surrogateType})`
      );
    }

    return loss;
  }
}
