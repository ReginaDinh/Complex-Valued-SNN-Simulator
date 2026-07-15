// Complex number helper class for TypeScript simulation
export interface Complex {
  re: number;
  im: number;
}

export const complex = {
  zero: (): Complex => ({ re: 0, im: 0 }),
  create: (re: number, im: number): Complex => ({ re, im }),
  fromPolar: (r: number, theta: number): Complex => ({
    re: r * Math.cos(theta),
    im: r * Math.sin(theta),
  }),
  add: (c1: Complex, c2: Complex): Complex => ({
    re: c1.re + c2.re,
    im: c1.im + c2.im,
  }),
  sub: (c1: Complex, c2: Complex): Complex => ({
    re: c1.re - c2.re,
    im: c1.im - c2.im,
  }),
  mul: (c1: Complex, c2: Complex): Complex => ({
    re: c1.re * c2.re - c1.im * c2.im,
    im: c1.re * c2.im + c1.im * c2.re,
  }),
  scale: (c: Complex, s: number): Complex => ({
    re: c.re * s,
    im: c.im * s,
  }),
  mag: (c: Complex): number => Math.sqrt(c.re * c.re + c.im * c.im),
  angle: (c: Complex): number => Math.atan2(c.im, c.re),
  conj: (c: Complex): Complex => ({ re: c.re, im: -c.im }),
};

// Complex-Valued Leaky Integrate-and-Fire Neuron
export class ComplexLIFNeuron {
  u: Complex = complex.zero(); // Membrane potential
  leak: Complex; // Complex decay beta
  threshold: number;
  resetMode: 'soft' | 'hard';
  
  constructor(leakMag: number, leakPhase: number, threshold: number, resetMode: 'soft' | 'hard' = 'soft') {
    this.leak = complex.fromPolar(leakMag, leakPhase);
    this.threshold = threshold;
    this.resetMode = resetMode;
  }

  reset() {
    this.u = complex.zero();
  }

  // Update step with complex-valued input current
  step(current: Complex): { spiked: boolean; spikeVal: Complex; uBefore: Complex; uAfter: Complex } {
    const uBefore = { ...this.u };
    
    // 1. Decay and Integrate: U(t) = beta * U(t-1) + I(t)
    const decayed = complex.mul(this.u, this.leak);
    this.u = complex.add(decayed, current);
    
    // 2. Threshold check: |U(t)| >= threshold
    const uMag = complex.mag(this.u);
    let spiked = false;
    let spikeVal = complex.zero();
    
    if (uMag >= this.threshold) {
      spiked = true;
      const angle = complex.angle(this.u);
      // Spike value is a unit complex number carrying the phase of the potential
      spikeVal = complex.fromPolar(1.0, angle);
      
      // 3. Reset
      if (this.resetMode === 'soft') {
        // Soft reset: U = U - V_th * e^{i * theta}
        const resetTerm = complex.fromPolar(this.threshold, angle);
        this.u = complex.sub(this.u, resetTerm);
      } else {
        // Hard reset: U = 0
        this.u = complex.zero();
      }
    }
    
    return {
      spiked,
      spikeVal,
      uBefore,
      uAfter: { ...this.u }
    };
  }
}

// Data generator for spectral classes
export interface SignalSample {
  inputSpikeSequence: Complex[][]; // [time][feature]
  target: number; // class index (0 or 1)
  frequencies: number[]; // frequencies generated
}

export function generateSpectralSignal(
  classType: number,
  timeSteps: number,
  numFeatures: number,
  freq0: number = 1.0,
  freq1: number = 4.5,
  phaseShift1: number = Math.PI / 2
): SignalSample {
  const sequence: Complex[][] = [];
  
  // Class 0: Low frequency component, in-phase signals
  // Class 1: High frequency component, phase-shifted signals
  const baseFreq = classType === 0 ? freq0 : freq1;
  const phaseShift = classType === 0 ? 0 : phaseShift1;
  
  for (let t = 0; t < timeSteps; t++) {
    const stepInputs: Complex[] = [];
    const tNormalized = t / timeSteps;
    
    for (let f = 0; f < numFeatures; f++) {
      // Frequency varies slightly per channel
      const freq = baseFreq * (1 + 0.1 * f);
      const amplitude = 0.8 + 0.4 * Math.sin(2 * Math.PI * tNormalized * 0.5);
      
      // Add phase modulation based on class and channel index
      const phase = 2 * Math.PI * freq * tNormalized + (f * phaseShift);
      
      // We generate complex values
      let val = complex.fromPolar(amplitude, phase);
      
      // Add complex Gaussian-like noise
      const noiseMag = 0.15;
      const noisePhase = Math.random() * 2 * Math.PI;
      const noise = complex.fromPolar(Math.random() * noiseMag, noisePhase);
      
      val = complex.add(val, noise);
      stepInputs.push(val);
    }
    sequence.push(stepInputs);
  }
  
  return {
    inputSpikeSequence: sequence,
    target: classType,
    frequencies: classType === 0 
      ? [freq0, freq0 * 1.1, freq0 * 1.2] 
      : [freq1, freq1 * 1.1, freq1 * 1.2],
  };
}

// A simple single-layer Complex SNN in TypeScript with learnable weights
export class TSComplexSNN {
  weights: Complex[][]; // [out_features][in_features]
  biases: Complex[]; // [out_features]
  inFeatures: number;
  outFeatures: number;
  neurons: ComplexLIFNeuron[];
  
  constructor(inFeatures: number, outFeatures: number, leakMag = 0.8, leakPhase = 0.15, threshold = 1.0, resetMode: 'soft' | 'hard' = 'soft') {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    
    // Initialize weights as complex numbers with small random magnitudes
    this.weights = [];
    for (let o = 0; o < outFeatures; o++) {
      const row: Complex[] = [];
      for (let i = 0; i < inFeatures; i++) {
        const mag = 0.2 + Math.random() * 0.3;
        const phase = (Math.random() - 0.5) * 2 * Math.PI;
        row.push(complex.fromPolar(mag, phase));
      }
      this.weights.push(row);
    }
    
    // Initialize biases
    this.biases = [];
    for (let o = 0; o < outFeatures; o++) {
      this.biases.push(complex.zero());
    }
    
    // Create neurons
    this.neurons = [];
    for (let o = 0; o < outFeatures; o++) {
      this.neurons.push(new ComplexLIFNeuron(leakMag, leakPhase, threshold, resetMode));
    }
  }

  reset() {
    this.neurons.forEach(n => n.reset());
  }

  // Forward pass through the network over time, returning outputs
  forward(xSeq: Complex[][]): {
    outputSpikeRates: number[];
    allOutputs: Complex[][]; // [time][class]
    allSpikes: boolean[][];  // [time][class]
    allPotentials: Complex[][]; // [time][class]
  } {
    this.reset();
    const timeSteps = xSeq.length;
    
    const allOutputs: Complex[][] = [];
    const allSpikes: boolean[][] = [];
    const allPotentials: Complex[][] = [];
    const spikeCounts = new Array(this.outFeatures).fill(0);
    
    for (let t = 0; t < timeSteps; t++) {
      const stepInputs = xSeq[t];
      const stepPotentials: Complex[] = [];
      const stepSpikes: boolean[] = [];
      const stepOuts: Complex[] = [];
      
      for (let o = 0; o < this.outFeatures; o++) {
        // Complex matrix multiplication: Current = sum(W_oi * X_i) + bias
        let current = this.biases[o];
        for (let i = 0; i < this.inFeatures; i++) {
          const product = complex.mul(this.weights[o][i], stepInputs[i]);
          current = complex.add(current, product);
        }
        
        // Feed into neuron
        const neuronResult = this.neurons[o].step(current);
        
        stepPotentials.push(neuronResult.uAfter);
        stepSpikes.push(neuronResult.spiked);
        stepOuts.push(neuronResult.spikeVal);
        
        if (neuronResult.spiked) {
          spikeCounts[o]++;
        }
      }
      
      allOutputs.push(stepOuts);
      allSpikes.push(stepSpikes);
      allPotentials.push(stepPotentials);
    }
    
    return {
      outputSpikeRates: spikeCounts.map(count => count / timeSteps),
      allOutputs,
      allSpikes,
      allPotentials
    };
  }

  // Evaluate loss (Cross-Entropy) on a batch
  evaluateLoss(sample: SignalSample): { loss: number; predictions: number[] } {
    const { outputSpikeRates } = this.forward(sample.inputSpikeSequence);
    
    // Apply softmax over firing rates (with stabilization)
    const maxRate = Math.max(...outputSpikeRates);
    const exps = outputSpikeRates.map(r => Math.exp(r - maxRate));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / (sumExps || 1e-9));
    
    const loss = -Math.log(probs[sample.target] + 1e-15);
    return {
      loss,
      predictions: probs
    };
  }

  // Train on a sample using numeric gradient descent (BPTT-equivalent)
  trainStep(sample: SignalSample, learningRate: number): number {
    const originalEval = this.evaluateLoss(sample);
    const originalLoss = originalEval.loss;
    
    const dW: Complex[][] = [];
    const epsilon = 1e-4;
    
    // Numeric gradient estimation for weights in the complex plane
    for (let o = 0; o < this.outFeatures; o++) {
      const rowGrad: Complex[] = [];
      for (let i = 0; i < this.inFeatures; i++) {
        // Gradient of Real weight
        const originalW = { ...this.weights[o][i] };
        
        this.weights[o][i] = { re: originalW.re + epsilon, im: originalW.im };
        const lossRealPertub = this.evaluateLoss(sample).loss;
        const gradRe = (lossRealPertub - originalLoss) / epsilon;
        
        // Gradient of Imaginary weight
        this.weights[o][i] = { re: originalW.re, im: originalW.im + epsilon };
        const lossImagPertub = this.evaluateLoss(sample).loss;
        const gradIm = (lossImagPertub - originalLoss) / epsilon;
        
        // Restore
        this.weights[o][i] = originalW;
        
        rowGrad.push({ re: gradRe, im: gradIm });
      }
      dW.push(rowGrad);
    }
    
    // Apply weight updates: W = W - lr * dW
    for (let o = 0; o < this.outFeatures; o++) {
      for (let i = 0; i < this.inFeatures; i++) {
        // Clip gradients to prevent explosion
        const maxGrad = 5.0;
        let gradRe = dW[o][i].re;
        let gradIm = dW[o][i].im;
        const gradMag = Math.sqrt(gradRe * gradRe + gradIm * gradIm);
        
        if (gradMag > maxGrad) {
          gradRe = (gradRe / gradMag) * maxGrad;
          gradIm = (gradIm / gradMag) * maxGrad;
        }
        
        this.weights[o][i].re -= learningRate * gradRe;
        this.weights[o][i].im -= learningRate * gradIm;
      }
    }
    
    return originalLoss;
  }
}
