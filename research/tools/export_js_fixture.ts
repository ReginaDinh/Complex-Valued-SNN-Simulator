// Exports reference outputs of the browser engine (src/simulationEngine.ts) so the Python library
// can be checked against it. Run from the repo root:
//   node --import tsx research/tools/export_js_fixture.ts > research/tests/fixtures/js_engine.json
import { TSComplexSNN, computeSurrogateDerivative, generateSpectralSignal, type SurrogateType } from '../../src/simulationEngine';
import { mulberry32 } from '../../src/utils/random';

function seededNet(seed: number, threshold: number, reset: 'soft' | 'hard'): TSComplexSNN {
  const orig = Math.random;
  Math.random = mulberry32(seed);
  try {
    return new TSComplexSNN(16, 2, 0.85, 0.15, threshold, reset, true, 'von_mises');
  } finally {
    Math.random = orig;
  }
}

const cases = [
  { name: 'soft_th1', threshold: 1.0, reset: 'soft' as const, netSeed: 42, dataSeed: 7, target: 1 },
  { name: 'hard_th06', threshold: 0.6, reset: 'hard' as const, netSeed: 3, dataSeed: 11, target: 0 },
].map(c => {
  const net = seededNet(c.netSeed, c.threshold, c.reset);
  const sample = generateSpectralSignal(c.target, 30, 16, 1.5, 4.5, 1.57, 0.15, mulberry32(c.dataSeed));
  const fwd = net.forward(sample.inputSpikeSequence, false);
  const ev = net.evaluateLoss(sample, false);
  return {
    ...c,
    leakMag: 0.85,
    leakPhase: 0.15,
    weights: net.weights.map(w => Array.from(w)),
    input: sample.inputSpikeSequence.map(s => Array.from(s)),
    preReset: fwd.preResetPotentials.map(p => Array.from(p)),
    postReset: fwd.allPotentials.map(p => Array.from(p)),
    spikes: fwd.spikeTrains,
    surrogateRates: fwd.surrogateRates,
    probs: ev.predictions,
  };
});

const grid = [0, 0.25, 0.5, 0.8, 0.95, 1.0, 1.05, 1.3, 2.0, 3.0];
const surrogate = (['von_mises', 'fast_sigmoid', 'atan'] as SurrogateType[]).flatMap(kind =>
  [0.6, 1.0, 1.5].map(th => ({ kind, threshold: th, m: grid, value: grid.map(m => computeSurrogateDerivative(m, th, kind, 2.5)) }))
);

process.stdout.write(JSON.stringify({ generatedBy: 'research/tools/export_js_fixture.ts', cases, surrogate }));
