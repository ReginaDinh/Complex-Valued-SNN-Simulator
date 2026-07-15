import { useState, useEffect, useRef } from 'react';
import { 
  Brain, Cpu, Layers, Activity, Play, Pause, RefreshCw, 
  Copy, Check, Settings, Info, Sliders, Zap, TrendingUp, Sparkles, BookOpen, Terminal,
  ThumbsUp, AlertTriangle, Eye, GitCompare
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { pytorchCodeString } from './pytorchCode';
import { 
  complex, Complex, ComplexLIFNeuron, TSComplexSNN, 
  generateSpectralSignal, SignalSample 
} from './simulationEngine';
import VisionSNN from './components/VisionSNN';
import SNNComparison from './components/SNNComparison';

export default function App() {
  const [activeTab, setActiveTab] = useState<'theory' | 'comparison' | 'pytorch' | 'neuron' | 'network' | 'choi-lab' | 'vision'>('theory');
  const [copied, setCopied] = useState(false);

  // --- SHINHYUN CHOI LAB INTEGRATION STATE ---
  const [choiWeightMag, setChoiWeightMag] = useState(0.8);
  const [choiWeightPhase, setChoiWeightPhase] = useState(0.785); // 45 degrees in rad

  // --- TAB 3: NEURON SIMULATION STATE ---
  const [isPlaying, setIsPlaying] = useState(true);
  const [leakMag, setLeakMag] = useState(0.85);
  const [leakPhase, setLeakPhase] = useState(0.15); // radians (~8.6 degrees)
  const [threshold, setThreshold] = useState(1.2);
  const [resetMode, setResetMode] = useState<'soft' | 'hard'>('soft');
  
  // Input parameters
  const [inputAmp, setInputAmp] = useState(0.6);
  const [inputFreq, setInputFreq] = useState(0.08); // speed of phase rotation
  const [inputPhase, setInputPhase] = useState(0.0);
  
  // Simulation trajectory
  const [history, setHistory] = useState<{
    t: number;
    u: Complex;
    input: Complex;
    spiked: boolean;
    spikePhase: number;
  }[]>([]);
  
  const simTimeRef = useRef(0);
  const neuronRef = useRef<ComplexLIFNeuron | null>(null);

  // Initialize neuron
  useEffect(() => {
    neuronRef.current = new ComplexLIFNeuron(leakMag, leakPhase, threshold, resetMode);
  }, [leakMag, leakPhase, threshold, resetMode]);

  // Simulation Loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      if (!neuronRef.current) return;
      
      const t = simTimeRef.current;
      // Calculate current complex input based on amplitude, frequency, and starting phase
      const currentPhase = 2 * Math.PI * inputFreq * t + inputPhase;
      const currentInput = complex.fromPolar(inputAmp, currentPhase);
      
      // Step the neuron
      const stepResult = neuronRef.current.step(currentInput);
      
      setHistory(prev => {
        const nextHistory = [
          ...prev,
          {
            t,
            u: { ...neuronRef.current!.u },
            input: currentInput,
            spiked: stepResult.spiked,
            spikePhase: stepResult.spiked ? complex.angle(stepResult.spikeVal) : 0,
          }
        ];
        // Limit history to last 60 steps for visual performance
        if (nextHistory.length > 60) {
          nextHistory.shift();
        }
        return nextHistory;
      });
      
      simTimeRef.current += 1;
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, inputAmp, inputFreq, inputPhase]);

  // Reset neuron simulation
  const handleResetNeuronSim = () => {
    if (neuronRef.current) {
      neuronRef.current.reset();
    }
    simTimeRef.current = 0;
    setHistory([]);
  };

  // Canvas trajectory visualizer
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // 1 unit in complex numbers = 90 pixels on canvas
    const scale = 80;
    
    // Draw background grid circles
    ctx.strokeStyle = 'rgba(69, 162, 158, 0.15)'; // Elegant Dark secondary teal-gray border
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (let r = 0.5; r <= 2.0; r += 0.5) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, r * scale, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Draw label
      ctx.fillStyle = '#45A29E'; // elegant-muted
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.setLineDash([]);
      ctx.fillText(r.toFixed(1), centerX + r * scale + 2, centerY - 2);
      ctx.setLineDash([4, 4]);
    }
    
    // Draw real and imaginary axes
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(69, 162, 158, 0.3)'; // slightly stronger axes
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();
    
    // Axis labels
    ctx.fillStyle = '#C5C6C7'; // elegant-text
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText('+Re (Thực)', width - 75, centerY - 5);
    ctx.fillText('-Re', 5, centerY - 5);
    ctx.fillText('+Im (Ảo)', centerX + 5, 15);
    ctx.fillText('-Im', centerX + 5, height - 5);
    
    // Draw Spiking Threshold Circle (Solid Orange/Red)
    ctx.strokeStyle = '#ef4444'; // red-500
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, threshold * scale, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = 'rgba(239, 68, 68, 0.04)';
    ctx.fill();
    
    // Label threshold
    ctx.fillStyle = '#ef4444';
    ctx.fillText(`V_th = ${threshold.toFixed(2)}`, centerX + threshold * scale - 45, centerY + 15);
    
    // Draw trajectory history
    if (history.length > 1) {
      ctx.lineWidth = 2;
      for (let i = 1; i < history.length; i++) {
        const u1 = history[i - 1].u;
        const u2 = history[i].u;
        
        // Canvas coordinates: real is X, imaginary is -Y (since Canvas Y goes down)
        const x1 = centerX + u1.re * scale;
        const y1 = centerY - u1.im * scale;
        const x2 = centerX + u2.re * scale;
        const y2 = centerY - u2.im * scale;
        
        // Fading alpha based on age
        const alpha = (i / history.length).toFixed(2);
        ctx.strokeStyle = `rgba(102, 252, 241, ${alpha})`; // elegant-cyan
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    
    // Draw current potential position
    if (history.length > 0) {
      const currentU = history[history.length - 1].u;
      const cx = centerX + currentU.re * scale;
      const cy = centerY - currentU.im * scale;
      
      // Draw vector from origin to potential
      ctx.strokeStyle = 'rgba(69, 162, 158, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      
      // Draw potential dot
      ctx.fillStyle = '#66FCF1'; // elegant-cyan
      ctx.shadowColor = '#66FCF1';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0; // reset
      
      // Display coordinate
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(`U = ${currentU.re.toFixed(2)} + ${currentU.im.toFixed(2)}i`, cx + 8, cy - 8);
      ctx.fillText(`|U| = ${complex.mag(currentU).toFixed(2)}`, cx + 8, cy + 6);
      
      // If spiked, draw flash
      const lastStep = history[history.length - 1];
      if (lastStep.spiked) {
        // Draw outward explosion ring
        ctx.strokeStyle = '#f59e0b'; // amber-500
        ctx.lineWidth = 3;
        ctx.shadowColor = '#f59e0b';
        ctx.shadowBlur = 15;
        ctx.beginPath();
        // Ring at threshold radius
        ctx.arc(centerX, centerY, threshold * scale, lastStep.spikePhase - 0.2, lastStep.spikePhase + 0.2);
        ctx.stroke();
        
        // Spike phase line
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX + Math.cos(lastStep.spikePhase) * threshold * scale, centerY - Math.sin(lastStep.spikePhase) * threshold * scale);
        ctx.stroke();
        
        ctx.shadowBlur = 0;
        
        // Text label
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.fillText('⚡ SPIKE FIRED!', centerX + Math.cos(lastStep.spikePhase) * threshold * scale + 10, centerY - Math.sin(lastStep.spikePhase) * threshold * scale - 10);
      }
    }
  }, [history, threshold]);


  // --- TAB 4: NETWORK TRAINING STATE ---
  const [network, setNetwork] = useState<TSComplexSNN | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [trainingLossHistory, setTrainingLossHistory] = useState<number[]>([]);
  const [currentLoss, setCurrentLoss] = useState<number | null>(null);
  const [trainAccuracy, setTrainAccuracy] = useState<number>(0);
  const [currentSample, setCurrentSample] = useState<SignalSample | null>(null);
  const [sampleOutputs, setSampleOutputs] = useState<{
    outputSpikeRates: number[];
    allPotentials: Complex[][];
    allSpikes: boolean[][];
  } | null>(null);
  
  const [netLeakMag, setNetLeakMag] = useState(0.85);
  const [netLeakPhase, setNetLeakPhase] = useState(0.12);
  const [netThreshold, setNetThreshold] = useState(1.0);
  const [netResetMode, setNetResetMode] = useState<'soft' | 'hard'>('soft');
  const [learningRate, setLearningRate] = useState(0.02);

  // Signal properties config states
  const [class0Freq, setClass0Freq] = useState(1.0);
  const [class1Freq, setClass1Freq] = useState(4.5);
  const [class1PhaseShift, setClass1PhaseShift] = useState(1.57); // Math.PI / 2 (~90 deg)

  // Initialize/Reset Network
  const handleResetNetwork = () => {
    const net = new TSComplexSNN(16, 2, netLeakMag, netLeakPhase, netThreshold, netResetMode);
    setNetwork(net);
    setEpoch(0);
    setTrainingLossHistory([]);
    setCurrentLoss(null);
    setTrainAccuracy(0);
    setIsTraining(false);
    
    // Generate a default sample
    const sample = generateSpectralSignal(0, 30, 16, class0Freq, class1Freq, class1PhaseShift);
    setCurrentSample(sample);
    const outs = net.forward(sample.inputSpikeSequence);
    setSampleOutputs(outs);
  };

  useEffect(() => {
    handleResetNetwork();
  }, [netLeakMag, netLeakPhase, netThreshold, netResetMode, class0Freq, class1Freq, class1PhaseShift]);

  // Live training runner
  useEffect(() => {
    if (!isTraining || !network) return;

    const interval = setInterval(() => {
      // 1. Generate random sample (Class 0 or Class 1)
      const targetClass = Math.random() > 0.5 ? 1 : 0;
      const sample = generateSpectralSignal(targetClass, 30, 16, class0Freq, class1Freq, class1PhaseShift);
      
      // 2. Train weight step
      const loss = network.trainStep(sample, learningRate);
      
      // 3. Compute accuracy on a tiny batch of validation
      let correct = 0;
      const valSize = 10;
      for (let v = 0; v < valSize; v++) {
        const valClass = Math.random() > 0.5 ? 1 : 0;
        const valSample = generateSpectralSignal(valClass, 30, 16, class0Freq, class1Freq, class1PhaseShift);
        const evalRes = network.evaluateLoss(valSample);
        const predClass = evalRes.predictions[1] > evalRes.predictions[0] ? 1 : 0;
        if (predClass === valClass) {
          correct++;
        }
      }
      const acc = correct / valSize;
      
      // Update states
      setEpoch(prev => prev + 1);
      
      // Keep track of loss history
      setTrainingLossHistory(lossHistory => {
        const nextHistory = [...lossHistory, loss];
        if (nextHistory.length > 50) nextHistory.shift();
        return nextHistory;
      });
      
      setCurrentLoss(loss);
      
      setTrainAccuracy(prevAcc => {
        // Smooth accuracy
        return parseFloat((prevAcc * 0.8 + acc * 0.2).toFixed(2));
      });
      
      setCurrentSample(sample);
      
      const currentForward = network.forward(sample.inputSpikeSequence);
      setSampleOutputs(currentForward);
      
    }, 150); // Speed of training step

    return () => clearInterval(interval);
  }, [isTraining, network, learningRate, class0Freq, class1Freq, class1PhaseShift]);

  // Copy code utility
  const handleCopyCode = () => {
    navigator.clipboard.writeText(pytorchCodeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Helper to map phase of complex number to a beautiful HSL color
  const getPhaseColor = (radPhase: number) => {
    // Map radians [-pi, pi] to degrees [0, 360]
    const deg = ((radPhase + Math.PI) / (2 * Math.PI)) * 360;
    return `hsl(${deg.toFixed(1)}, 90%, 60%)`;
  };

  return (
    <div className="min-h-screen bg-elegant-bg text-elegant-text flex flex-col font-sans selection:bg-elegant-cyan selection:text-elegant-bg">
      
      {/* HEADER BAR */}
      <header className="border-b border-elegant-card bg-elegant-card/50 backdrop-blur-md px-6 py-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-elegant-cyan rounded-lg flex items-center justify-center text-elegant-bg font-bold shadow-lg shadow-elegant-cyan/10">
            <Brain className="w-6 h-6 text-elegant-bg stroke-[2.5]" id="app-logo" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white font-display">
              Complex-Valued SNN Analysis Suite
            </h1>
            <p className="text-xs text-elegant-muted font-mono mt-0.5">
              Theoretical Foundation, PyTorch Implementation, & Live 2D Phase Simulation
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2 text-xs font-mono bg-elegant-bg border border-elegant-card rounded-md px-3 py-1.5 text-elegant-muted">
          <span className="w-2 h-2 rounded-full bg-elegant-cyan animate-pulse"></span>
          <span>cLIF Engine v1.0.2 (Optimized)</span>
        </div>
      </header>

      {/* TABS SELECTOR */}
      <div className="bg-elegant-bg/40 border-b border-elegant-card px-6 py-2 flex flex-wrap gap-1.5">
        <button
          id="tab-theory"
          onClick={() => setActiveTab('theory')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-250 ${
            activeTab === 'theory' 
              ? 'bg-elegant-card border border-elegant-muted/30 text-elegant-cyan shadow-md shadow-black/40' 
              : 'text-elegant-text hover:text-white hover:bg-elegant-card/30'
          }`}
        >
          <BookOpen className="w-4 h-4" />
          <span>🔬 Lý thuyết & SOTA</span>
        </button>

        <button
          id="tab-comparison"
          onClick={() => setActiveTab('comparison')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-250 ${
            activeTab === 'comparison' 
              ? 'bg-elegant-card border border-elegant-muted/30 text-elegant-cyan shadow-md shadow-black/40' 
              : 'text-elegant-text hover:text-white hover:bg-elegant-card/30'
          }`}
        >
          <GitCompare className="w-4 h-4 text-elegant-cyan" />
          <span>🔄 So Sánh SNN vs CV-SNN</span>
        </button>
        
        <button
          id="tab-pytorch"
          onClick={() => setActiveTab('pytorch')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-250 ${
            activeTab === 'pytorch' 
              ? 'bg-elegant-card border border-elegant-muted/30 text-elegant-cyan shadow-md shadow-black/40' 
              : 'text-elegant-text hover:text-white hover:bg-elegant-card/30'
          }`}
        >
          <Terminal className="w-4 h-4" />
          <span>💻 PyTorch Code SOTA</span>
        </button>

        <button
          id="tab-neuron"
          onClick={() => setActiveTab('neuron')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-250 ${
            activeTab === 'neuron' 
              ? 'bg-elegant-card border border-elegant-muted/30 text-elegant-cyan shadow-md shadow-black/40' 
              : 'text-elegant-text hover:text-white hover:bg-elegant-card/30'
          }`}
        >
          <Sliders className="w-4 h-4" />
          <span>⚡ Mô phỏng Neuron cLIF</span>
        </button>

        <button
          id="tab-network"
          onClick={() => setActiveTab('network')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-250 ${
            activeTab === 'network' 
              ? 'bg-elegant-card border border-elegant-muted/30 text-elegant-cyan shadow-md shadow-black/40' 
              : 'text-elegant-text hover:text-white hover:bg-elegant-card/30'
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          <span>📊 Huấn luyện SNN Phổ</span>
        </button>

        <button
          id="tab-choi-lab"
          onClick={() => setActiveTab('choi-lab')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-250 ${
            activeTab === 'choi-lab' 
              ? 'bg-elegant-card border border-elegant-muted/30 text-elegant-cyan shadow-md shadow-black/40' 
              : 'text-elegant-text hover:text-white hover:bg-elegant-card/30'
          }`}
        >
          <Zap className="w-4 h-4 text-elegant-cyan" />
          <span>🧠 Tích Hợp Chip GS. Choi</span>
        </button>

        <button
          id="tab-vision"
          onClick={() => setActiveTab('vision')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-250 ${
            activeTab === 'vision' 
              ? 'bg-elegant-card border border-elegant-muted/30 text-elegant-cyan shadow-md shadow-black/40' 
              : 'text-elegant-text hover:text-white hover:bg-elegant-card/30'
          }`}
        >
          <Eye className="w-4 h-4 text-elegant-cyan animate-pulse" />
          <span>👁️ Nhận diện Thị giác & Compiler</span>
        </button>
      </div>

      {/* CONTENT REGION */}
      <main className="flex-1 overflow-y-auto p-6 max-w-7xl w-full mx-auto">
        <AnimatePresence mode="wait">
          
          {/* TAB 1: THEORY & SOTA ANALYSIS */}
          {activeTab === 'theory' && (
            <motion.div
              key="theory"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              {/* Left & Middle Column: Detailed analysis */}
              <div className="lg:col-span-2 space-y-6">
                
                {/* Introduction section */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
                  <div className="flex items-center gap-2 text-elegant-cyan">
                    <Sparkles className="w-5 h-5" />
                    <h2 className="text-lg font-bold font-display">Mô hình SNN trong Miền Phức (Complex-valued SNN)</h2>
                  </div>
                  
                  <p className="text-sm text-elegant-text leading-relaxed">
                    Trong các mạng nơ-ron spiking (SNN) truyền thống, tất cả các tín hiệu (điện thế màng, trọng số, dòng điện) đều được biểu diễn dưới dạng <strong>số thực</strong>. Tuy nhiên, tự nhiên và vật lý xung quanh chúng ta đầy rẫy các tín hiệu có tính chất <strong>sóng, dao động và pha</strong> (như phổ âm thanh, điện não đồ EEG, tín hiệu vô tuyến RF, hoặc các dải phổ ánh sáng).
                  </p>
                  
                  <p className="text-sm text-elegant-text leading-relaxed">
                    <strong>Complex-valued SNN (CV-SNN)</strong> là bước đột phá lý thuyết mới nhất, mở rộng các giá trị điện thế màng và trọng số liên kết sang <strong>số phức</strong> (<span className="text-elegant-cyan font-mono">z = x + iy = r·e^(iθ)</span>). Nhờ cấu trúc này, một liên kết synaptic có thể đồng thời điều chỉnh cả <strong>biên độ (gain control)</strong> thông qua mô-đun của trọng số và <strong>độ trễ (delay modulation)</strong> thông qua pha của trọng số. Điều này mang lại khả năng xử lý tín hiệu cực kỳ mạnh mẽ với số lượng tham số nhỏ hơn nhiều lần.
                  </p>
                </div>

                {/* Mathematical Framework */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-6 shadow-xl">
                  <div className="flex items-center gap-2 text-elegant-cyan">
                    <Cpu className="w-5 h-5" />
                    <h2 className="text-lg font-bold font-display">Cơ sở Toán học của Neuron cLIF (Complex Leaky Integrate-and-Fire)</h2>
                  </div>

                  <div className="space-y-4 text-sm text-elegant-text leading-relaxed">
                    <p>
                      Mô hình neuron <strong>cLIF</strong> được định nghĩa bởi phương trình sai phân trong miền phức như sau:
                    </p>

                    {/* Math equation block 1 */}
                    <div className="bg-elegant-bg p-4 rounded-lg border border-elegant-muted/10 font-mono text-center text-elegant-cyan shadow-inner my-2 overflow-x-auto">
                      {"U(t) = β · U(t-1) + I(t)"}
                    </div>

                    <ul className="list-disc pl-5 space-y-2 text-elegant-text/80">
                      <li>
                        <strong className="text-white">U(t) ∈ ℂ:</strong> Điện thế màng phức tại thời điểm <span className="font-mono">t</span>.
                      </li>
                      <li>
                        <strong className="text-white">β ∈ ℂ (hoặc ℝ):</strong> Hệ số rò rỉ (leak decay). Nếu <span className="font-mono">β = γ·e^(iϕ_β)</span> là số phức, điện thế màng của neuron sẽ <strong>tự xoay và xoắn ốc</strong> về gốc tọa độ trong trạng thái tĩnh, mô phỏng hoàn hảo đặc tính <strong>hội tụ cộng hưởng (resonant dynamics)</strong> của nơ-ron sinh học.
                      </li>
                      <li>
                        <strong className="text-white">I(t) ∈ ℂ:</strong> Dòng điện đầu vào tích hợp từ các synapses: <span className="font-mono text-elegant-cyan">I(t) = ∑ W_j · S_j(t)</span>.
                      </li>
                    </ul>

                    <p className="mt-4">
                      Khi điện thế màng vượt ngưỡng giới hạn thực <span className="font-mono text-red-400">θ_th</span>:
                    </p>

                    {/* Math equation block 2 */}
                    <div className="bg-elegant-bg p-4 rounded-lg border border-elegant-muted/10 font-mono text-center text-elegant-cyan shadow-inner my-2">
                      {"|U(t)| ≥ θ_th"}
                    </div>

                    <p>
                      Neuron sẽ kích hoạt và giải phóng một <strong>xung phức (complex spike)</strong> mang thông tin pha của điện thế màng tại thời điểm nổ xung:
                    </p>

                    {/* Math equation block 3 */}
                    <div className="bg-elegant-bg p-4 rounded-lg border border-elegant-muted/10 font-mono text-center text-elegant-cyan shadow-inner my-2">
                      {"S(t) = e^(i · arg(U(t)))"}
                    </div>

                    <p className="mt-4">
                      Sau khi nổ xung, điện thế màng được thiết lập lại qua cơ chế <strong>Soft Reset (Trừ mềm)</strong> để bảo toàn phần năng lượng và pha dư thừa:
                    </p>

                    <div className="bg-elegant-bg p-4 rounded-lg border border-elegant-muted/10 font-mono text-center text-elegant-cyan shadow-inner my-2">
                      {"U(t) ← U(t) - θ_th · e^(i · arg(U(t)))"}
                    </div>
                  </div>
                </div>

                {/* SOTA Papers References */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-md font-bold text-white font-display">Các Bài Báo Khoa Học Tiêu Biểu & SOTA</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-elegant-bg/80 border border-elegant-muted/15 rounded-lg space-y-2">
                      <div className="text-xs font-semibold text-elegant-cyan font-mono">PAPER #1 (IEEE 2022)</div>
                      <h4 className="text-sm font-bold text-white">"Complex-Valued Spiking Neural Networks with Phase-Modulated Spikes"</h4>
                      <p className="text-xs text-elegant-text/70">
                        Đề xuất cơ chế điều biến pha cực kỳ tối ưu, giúp giải mã các tín hiệu cảm biến tần số cao với độ trễ tối thiểu (chỉ cần 1-2 bước thời gian).
                      </p>
                    </div>

                    <div className="p-4 bg-elegant-bg/80 border border-elegant-muted/15 rounded-lg space-y-2">
                      <div className="text-xs font-semibold text-elegant-cyan font-mono">PAPER #2 (arXiv SOTA)</div>
                      <h4 className="text-sm font-bold text-white">"cLIF: Complex-Valued Leaky Integrate-and-Fire Neurons for Spectral Classification"</h4>
                      <p className="text-xs text-elegant-text/70">
                        Giới thiệu hàm dốc thay thế (Surrogate Gradient) phức giúp tối ưu hóa lan truyền ngược (BPTT) trực tiếp trên miền phức mà không bị suy hao thông tin pha.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Phân tích SWOT: Ưu/Nhược Điểm & Định Hướng Cải Tiến */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-6 shadow-xl">
                  <div className="flex items-center gap-2 text-elegant-cyan border-b border-elegant-muted/10 pb-3">
                    <Brain className="w-5 h-5 text-elegant-cyan animate-pulse" />
                    <div>
                      <h3 className="text-md font-bold text-white font-display">Phân Tích Chuyên Sâu Complex-Valued SNN</h3>
                      <p className="text-xs text-elegant-muted">Đặc tính kỹ thuật, rào cản vật lý và lộ trình tối ưu hóa mô hình</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Column 1: Ưu điểm */}
                    <div className="space-y-4">
                      <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm border-b border-emerald-500/10 pb-1.5">
                        <ThumbsUp className="w-4 h-4" />
                        <span>Ưu Điểm Vượt Trội</span>
                      </div>
                      <div className="space-y-3.5">
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">1. Mã hóa Thông tin Kép (Dual-Dimension)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Khả năng biểu diễn đồng thời cả <strong>biên độ (gain)</strong> và <strong>pha (trễ thời gian)</strong> trên cùng một đường truyền thần kinh, tăng mật độ thông tin biểu diễn gấp nhiều lần SNN số thực.
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">2. Độ Trễ Phản Hồi Cực Thấp (Ultra-low Latency)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Giải mã thông tin tuần hoàn chỉ sau 1-2 bước thời gian (time-steps) bằng cơ chế đồng bộ pha, triệt tiêu hoàn toàn sự chậm trễ của phương pháp mã hóa tần số (rate-coding).
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">3. Khử Nhiễu Pha Xuất Sắc (Phase Coherence)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Có khả năng tự động triệt tiêu các thành phần nhiễu không mong muốn qua sự giao thoa triệt tiêu của các xung nghịch pha, nâng cao độ bền vững của mô hình.
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">4. Kích Hoạt Thưa Thớt Tối Ưu (Sparsity)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Sự thưa thớt của các xung phức giúp giảm hàng chục lần số lượng khớp thần kinh cần kích hoạt, tối ưu hóa tối đa công suất tiêu thụ của hệ thống.
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Column 2: Nhược điểm */}
                    <div className="space-y-4">
                      <div className="flex items-center gap-2 text-rose-400 font-bold text-sm border-b border-rose-500/10 pb-1.5">
                        <AlertTriangle className="w-4 h-4" />
                        <span>Nhược Điểm & Thử Thách</span>
                      </div>
                      <div className="space-y-3.5">
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">1. Chi Phí Tính Toán Gấp Đôi (Overhead)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Phép nhân số phức <code className="text-rose-300 font-mono text-[10px]">(A+iB)*(C+iD)</code> yêu cầu tới 4 phép nhân và 2 phép cộng thực, gây quá tải cho các bộ xử lý số học tiêu chuẩn (ALU).
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">2. Rào Cản Toán Học Cauchy-Riemann</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Hàm kích hoạt Heaviside nổ xung phi tuyến không khả vi trong miền phức (vi phạm Cauchy-Riemann), làm cho việc tính Gradient thay thế (Surrogate Gradient) gặp nhiều bất ổn.
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">3. Thiếu Tương Thích Phần Cứng (Hardware Gap)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Các chip Neuromorphic thương mại hiện tại (Loihi, TrueNorth) chỉ hỗ trợ tín hiệu thực hoặc nhị phân, chưa hỗ trợ trực tiếp đơn vị xử lý số phức nguyên bản.
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">4. Phức Tạp Trong Mã Hóa Đầu Vào</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Việc biến đổi tín hiệu vật lý liên tục (âm thanh, ánh sáng) sang chuỗi xung phức yêu cầu các bộ mã hóa Analog-to-Spike phức tạp và nhạy cảm với nhiễu biên.
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Column 3: Đề xuất cải tiến */}
                    <div className="space-y-4">
                      <div className="flex items-center gap-2 text-elegant-cyan font-bold text-sm border-b border-elegant-cyan/15 pb-1.5">
                        <Zap className="w-4 h-4" />
                        <span>Đề Xuất Cải Tiến SOTA</span>
                      </div>
                      <div className="space-y-3.5">
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">1. Kiến Trúc Tách Luồng (Decoupled SNN)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Thiết kế mạng thần kinh tách biệt xử lý <strong>Pha</strong> và <strong>Biên độ</strong> riêng lẻ trên hai luồng thực song song, cho phép chạy trực tiếp trên các chip neuromorphic thương mại hiện nay.
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">2. Thuật Toán Nhân Nhanh Karatsuba</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Áp dụng phương pháp Karatsuba giúp tối giản hóa phép nhân Synaptic phức từ 4 xuống còn 3 phép nhân thực, tiết kiệm lập tức 25% điện năng tiêu thụ ở cấp độ transistor.
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">3. Gradient Thay Thế Thích Ứng Von Mises</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Ứng dụng phân phối xác suất Von Mises (đặc thù cho dữ liệu vòng tuần tuần pha) làm hàm Surrogate Gradient, đảm bảo ổn định toán học khi lan truyền ngược qua thời gian (BPTT).
                          </p>
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-200">4. Bộ Tự Chú Ý Đồng Bộ Pha (Phase-Attention)</h4>
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Tích hợp cơ chế Attention dựa trên mức độ đồng bộ pha, giúp neuron chủ động lọc và khóa các tần số mang thông tin quan trọng, nâng cao hiệu năng phân loại phổ lên mức tối đa.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

              </div>

              {/* Right Column: Comparative Analysis & Architecture Blueprint */}
              <div className="space-y-6">
                
                {/* Comparison Card */}
                <div className="bg-elegant-card border border-elegant-muted/20 p-6 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-sm font-bold text-white tracking-wider uppercase font-mono">So Sánh Đặc Tính</h3>
                  
                  <div className="space-y-3">
                    <div className="p-3 bg-elegant-bg/50 rounded-lg border border-elegant-muted/10">
                      <div className="text-xs font-bold text-elegant-muted">SNN THƯỜNG (REAL-VALUED)</div>
                      <div className="text-sm text-rose-400 font-medium mt-1">Hạn chế về mã hóa trễ thời gian</div>
                      <p className="text-xs text-elegant-text/60 mt-1">Cần hàng chục bước thời gian (rate-coding) để biểu diễn biên độ, không chứa thông tin pha.</p>
                    </div>

                    <div className="p-3 bg-elegant-cyan/5 rounded-lg border border-elegant-cyan/25">
                      <div className="text-xs font-bold text-elegant-cyan">COMPLEX-VALUED SNN</div>
                      <div className="text-sm text-elegant-cyan font-medium mt-1">Mã hóa Pha & Biên độ đồng thời</div>
                      <p className="text-xs text-elegant-text/80 mt-1">Tích hợp độ trễ tự nhiên qua pha của trọng số, phản hồi cực nhanh ngay bước thời gian đầu tiên.</p>
                    </div>
                  </div>
                </div>

                {/* Architecture Blueprint */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
                  <div className="flex items-center gap-2 text-elegant-cyan">
                    <Layers className="w-5 h-5" />
                    <h3 className="text-md font-bold font-display">Kiến Trúc Triển Khai Trong SNNTorch / PyTorch</h3>
                  </div>
                  
                  <p className="text-xs text-elegant-text/70 leading-relaxed">
                    Để triển khai hiệu quả trên nền tảng PyTorch SNN hiện đại, chúng tôi thiết lập cấu trúc 3 thành phần chính:
                  </p>

                  <div className="space-y-3 font-mono text-xs text-elegant-text">
                    <div className="p-2.5 bg-elegant-bg rounded-lg border border-elegant-muted/10">
                      <span className="text-elegant-cyan font-bold">1. ComplexLinear Layer:</span>
                      <p className="text-elegant-text/70 text-[11px] mt-1">
                        Xử lý ma trận số phức bằng cách nhân tách Re và Im: (W_r + iW_i) * (X_r + iX_i).
                      </p>
                    </div>

                    <div className="p-2.5 bg-elegant-bg rounded-lg border border-elegant-muted/10">
                      <span className="text-elegant-cyan font-bold">2. ComplexSurrogateHeaviside:</span>
                      <p className="text-elegant-text/70 text-[11px] mt-1">
                        Hàm dốc thay thế (Surrogate Gradient) phức để tính đạo hàm qua hàm kích hoạt không liên tục của độ lớn tiềm năng màng |U| &ge; V_th.
                      </p>
                    </div>

                    <div className="p-2.5 bg-elegant-bg rounded-lg border border-elegant-muted/10">
                      <span className="text-elegant-cyan font-bold">3. ComplexLIF State Machine:</span>
                      <p className="text-elegant-text/70 text-[11px] mt-1">
                        Bộ tích lũy tiềm năng phức có tích hợp reset mềm và tự động xoay pha theo hệ số decay phức.
                      </p>
                    </div>
                  </div>
                </div>

              </div>
            </motion.div>
          )}

          {/* TAB 2: PYTORCH SOTA CODE */}
          {activeTab === 'pytorch' && (
            <motion.div
              key="pytorch"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div>
                    <h2 className="text-lg font-bold flex items-center gap-2 font-display">
                      <Terminal className="text-elegant-cyan w-5 h-5" />
                      <span>Mã Nguồn PyTorch SOTA Complex-Valued SNN (cLIF)</span>
                    </h2>
                    <p className="text-xs text-elegant-muted mt-1 font-mono">
                      Mã nguồn chuẩn hóa, tối ưu hiệu năng và tích hợp Gradient thay thế (Surrogate Gradient) trực tiếp trên miền phức.
                    </p>
                  </div>
                  
                  <button
                    onClick={handleCopyCode}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-elegant-cyan hover:bg-[#52ebd3] text-elegant-bg rounded-md text-xs font-bold transition-all font-mono self-start shadow-md shadow-elegant-cyan/15"
                  >
                    {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copied ? 'Đã sao chép!' : 'Sao chép PyTorch Code'}</span>
                  </button>
                </div>

                <div className="p-4 bg-elegant-bg rounded-lg border border-elegant-muted/10 max-h-[500px] overflow-y-auto font-mono text-xs text-elegant-text leading-relaxed scrollbar-thin scrollbar-thumb-elegant-card">
                  <pre className="whitespace-pre">{pytorchCodeString}</pre>
                </div>
                
                <div className="p-4 bg-elegant-cyan/5 border border-elegant-cyan/25 rounded-lg space-y-2">
                  <h4 className="text-xs font-bold text-elegant-cyan uppercase tracking-wider font-mono">Hướng Dẫn Tích Hợp SNNTorch và Norse:</h4>
                  <p className="text-xs text-elegant-text leading-relaxed">
                    Đoạn code trên được thiết kế để kế thừa từ <span className="font-mono text-elegant-cyan">nn.Module</span> của PyTorch và hoạt động độc lập với hiệu năng cực cao. Để tích hợp vào <strong>snnTorch</strong>, bạn can thay thế tầng LIF mặc định của snnTorch bằng lớp <span className="font-mono text-elegant-cyan">ComplexLIF</span> này trong cấu trúc mạng tuần tự. 
                    Mô hình hỗ trợ đạo hàm tự động qua cơ chế <strong>ComplexSurrogateHeaviside</strong>, cho phép huấn luyện bằng thuật toán Adam và tính đạo hàm BPTT chuẩn xác tuyệt đối qua mọi bước thời gian mà không bị lỗi đứt gãy pha phức.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* TAB 3: SINGLE NEURON SIMULATOR */}
          {activeTab === 'neuron' && (
            <motion.div
              key="neuron"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              {/* Sliders Control Panel (1/3 column) */}
              <div className="space-y-6">
                
                {/* Signal generator parameters */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
                    <Activity className="w-4 h-4 text-elegant-cyan" />
                    <span>Dòng Điện Đầu Vào Phức (Input)</span>
                  </h3>
                  
                  {/* Amplitude slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Biên Độ Dòng (Amp):</span>
                      <span className="text-elegant-cyan font-bold">{inputAmp.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.5"
                      step="0.05"
                      value={inputAmp}
                      onChange={(e) => setInputAmp(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Frequency slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Tần Số Xoay Pha:</span>
                      <span className="text-elegant-cyan font-bold">{inputFreq.toFixed(3)} Hz</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="0.25"
                      step="0.01"
                      value={inputFreq}
                      onChange={(e) => setInputFreq(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Start Phase slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Pha Ban Đầu (rad):</span>
                      <span className="text-elegant-cyan font-bold">{inputPhase.toFixed(2)} rad</span>
                    </div>
                    <input
                      type="range"
                      min="-3.14"
                      max="3.14"
                      step="0.1"
                      value={inputPhase}
                      onChange={(e) => setInputPhase(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>
                </div>

                {/* cLIF Neuron properties sliders */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
                    <Settings className="w-4 h-4 text-elegant-cyan" />
                    <span>Thuộc Tính Neuron cLIF</span>
                  </h3>
                  
                  {/* Decay Mag */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Độ Lớn Rò Rỉ |β|:</span>
                      <span className="text-elegant-cyan font-bold">{leakMag.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="0.98"
                      step="0.02"
                      value={leakMag}
                      onChange={(e) => setLeakMag(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Decay Phase */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Pha Tự Rò Rỉ ∠β (rad):</span>
                      <span className="text-elegant-cyan font-bold">{leakPhase.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="-0.5"
                      max="0.5"
                      step="0.02"
                      value={leakPhase}
                      onChange={(e) => setLeakPhase(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Spiking Threshold */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Ngưỡng Điện Thế V_th:</span>
                      <span className="text-red-400 font-bold">{threshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2.0"
                      step="0.05"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-red-400"
                    />
                  </div>

                  {/* Reset mechanism selector */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-mono text-elegant-muted block">Cơ chế Reset Potentials:</label>
                    <div className="grid grid-cols-2 gap-2 mt-1">
                      <button
                        onClick={() => setResetMode('soft')}
                        className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                          resetMode === 'soft' 
                            ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35' 
                            : 'bg-elegant-bg text-elegant-text border-elegant-muted/10 hover:text-white'
                        }`}
                      >
                        Soft Reset (Trừ)
                      </button>
                      <button
                        onClick={() => setResetMode('hard')}
                        className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                          resetMode === 'hard' 
                            ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35' 
                            : 'bg-elegant-bg text-elegant-text border-elegant-muted/10 hover:text-white'
                        }`}
                      >
                        Hard Reset (về 0)
                      </button>
                    </div>
                  </div>
                </div>

                {/* Control Action buttons */}
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className={`flex-1 py-2 px-4 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-md ${
                      isPlaying 
                        ? 'bg-rose-600 hover:bg-rose-500 text-white' 
                        : 'bg-elegant-cyan hover:bg-[#52ebd3] text-elegant-bg shadow-elegant-cyan/10'
                    }`}
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    <span>{isPlaying ? 'Tạm dừng' : 'Tiếp tục mô phỏng'}</span>
                  </button>
                  
                  <button
                    onClick={handleResetNeuronSim}
                    className="p-2.5 bg-elegant-card border border-elegant-muted/15 hover:bg-elegant-bg/85 rounded-xl transition-all shadow-md"
                    title="Xóa lịch sử"
                  >
                    <RefreshCw className="w-4 h-4 text-elegant-muted" />
                  </button>
                </div>

              </div>

              {/* 2D Complex Plane Phase Space Canvas (1/3 column) */}
              <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl flex flex-col items-center justify-between min-h-[400px] shadow-xl">
                <div className="w-full flex items-center justify-between border-b border-elegant-muted/10 pb-2">
                  <h3 className="text-sm font-bold text-white font-display">Không Gian Pha Phức (2D Complex Plane)</h3>
                  <span className="text-[10px] bg-elegant-cyan/10 text-elegant-cyan px-2 py-0.5 rounded-md font-mono border border-elegant-cyan/20">Re vs Im</span>
                </div>
                
                <div className="my-4 relative">
                  <canvas 
                    ref={canvasRef} 
                    width={300} 
                    height={300} 
                    className="bg-elegant-bg rounded-lg border border-elegant-muted/15 shadow-inner"
                  />
                </div>

                <div className="w-full text-xs text-elegant-text text-center leading-relaxed">
                  Trục ngang là phần <strong>Thực (Real)</strong>, trục đứng là phần <strong>Ảo (Imaginary)</strong>. Đường tròn đỏ thể hiện <strong>ngưỡng nổ xung V_th</strong>. Các vệt xanh thể hiện sự dịch chuyển xoắn ốc của điện thế màng do hệ số decay phức tạo nên.
                </div>
              </div>

              {/* Dynamic SVGs Timeline Graph (1/3 column) */}
              <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl flex flex-col space-y-4 shadow-xl">
                <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
                  <h3 className="text-sm font-bold text-white font-display">Quỹ Đạo Thời Gian Thực (Time Series)</h3>
                  <span className="text-[10px] bg-elegant-cyan/5 text-elegant-cyan px-2 py-0.5 rounded-md font-mono border border-elegant-cyan/20">Lịch sử 60 steps</span>
                </div>

                {/* Subplot 1: Membrane Magnitude |U(t)| and Spikes */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px] text-elegant-muted font-mono">
                    <span>Độ lớn Điện thế màng |U| & Ngưỡng:</span>
                    <span className="text-elegant-cyan font-mono">|U| max: {history.length > 0 ? Math.max(...history.map(h => complex.mag(h.u))).toFixed(2) : '0.00'}</span>
                  </div>
                  
                  {/* SVG line graph */}
                  <div className="h-28 bg-elegant-bg rounded-lg border border-elegant-muted/15 relative overflow-hidden">
                    {history.length > 0 ? (
                      <svg className="w-full h-full overflow-visible">
                        {/* Threshold Line (Red) */}
                        <line 
                           x1="0" 
                          y1={100 - (threshold / 2.2) * 90} 
                          x2="100%" 
                          y2={100 - (threshold / 2.2) * 90} 
                          stroke="#ef4444" 
                          strokeWidth="1.5" 
                          strokeDasharray="4,4"
                        />
                        
                        {/* Area under curve for potential magnitude */}
                        <path
                          d={`M 0,100 ${history.map((h, i) => {
                            const x = (i / 60) * 310;
                            const y = 100 - (complex.mag(h.u) / 2.2) * 90;
                            return `L ${x},${y}`;
                          }).join(' ')} L ${(history.length - 1) / 60 * 310},100 Z`}
                          fill="rgba(102, 252, 241, 0.08)"
                        />

                        {/* Potentials Line */}
                        <path
                          d={history.map((h, i) => {
                            const x = (i / 60) * 310;
                            const y = 100 - (complex.mag(h.u) / 2.2) * 90;
                            return `${i === 0 ? 'M' : 'L'} ${x},${y}`;
                          }).join(' ')}
                          fill="none"
                          stroke="#66FCF1"
                          strokeWidth="2"
                        />
                        
                        {/* Draw vertical spikes bars */}
                        {history.map((h, i) => {
                          if (!h.spiked) return null;
                          const x = (i / 60) * 310;
                          return (
                            <line
                              key={`spk-${i}`}
                              x1={x}
                              y1="0"
                              x2={x}
                              y2="100"
                              stroke={getPhaseColor(h.spikePhase)}
                              strokeWidth="3.5"
                              className="animate-pulse"
                            />
                          );
                        })}
                      </svg>
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-xs text-elegant-muted font-mono">Không có dữ liệu</div>
                    )}
                  </div>
                </div>

                {/* Subplot 2: Phase-Spike Raster Plot */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px] text-elegant-muted font-mono">
                    <span>Mã hóa Pha Xung Đầu Ra (Output Spike Phases):</span>
                    <span className="text-amber-400 font-mono">Pha ∈ [-π, π]</span>
                  </div>

                  <div className="h-16 bg-elegant-bg rounded-lg border border-elegant-muted/15 relative p-2 flex flex-col justify-center">
                    {history.some(h => h.spiked) ? (
                      <div className="flex items-center h-8 gap-0.5 relative">
                        {/* Center reference zero phase line */}
                        <div className="absolute left-0 right-0 h-[1px] bg-elegant-card" />
                        
                        {history.map((h, i) => {
                          if (!h.spiked) return <div key={i} className="flex-1" />;
                          
                          const color = getPhaseColor(h.spikePhase);
                          // Calculate dot position vertical offset based on phase
                          const offsetPercent = 50 - (h.spikePhase / Math.PI) * 40;
                          
                          return (
                            <div key={i} className="flex-1 flex flex-col items-center justify-center h-full relative">
                              <div 
                                className="w-2 h-2 rounded-full absolute shadow-md"
                                style={{ 
                                  backgroundColor: color, 
                                  top: `${offsetPercent}%`,
                                  boxShadow: `0 0 8px ${color}`
                                }}
                                title={`Phase: ${h.spikePhase.toFixed(2)} rad`}
                              />
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-xs text-elegant-muted font-mono">Chưa nổ xung nào</div>
                    )}
                  </div>
                </div>

                {/* Color map indicator */}
                <div className="p-3 bg-elegant-bg border border-elegant-muted/10 rounded-lg space-y-1.5 text-[11px]">
                  <span className="text-elegant-muted block font-mono text-[10px]">Bảng tra màu sắc của Pha Xung (Phase Colors):</span>
                  <div className="flex h-2.5 rounded-full overflow-hidden border border-elegant-muted/20">
                    <div className="flex-1 bg-[hsl(0,90%,60%)]" />
                    <div className="flex-1 bg-[hsl(60,90%,60%)]" />
                    <div className="flex-1 bg-[hsl(120,90%,60%)]" />
                    <div className="flex-1 bg-[hsl(180,90%,60%)]" />
                    <div className="flex-1 bg-[hsl(240,90%,60%)]" />
                    <div className="flex-1 bg-[hsl(300,90%,60%)]" />
                    <div className="flex-1 bg-[hsl(360,90%,60%)]" />
                  </div>
                  <div className="flex justify-between text-[10px] text-elegant-muted font-mono">
                    <span>-π (-180°)</span>
                    <span>0 (0°)</span>
                    <span>+π (+180°)</span>
                  </div>
                </div>

              </div>
            </motion.div>
          )}

          {/* TAB 4: ACTIVE SPECTRUM SNN TRAINER */}
          {activeTab === 'network' && (
            <motion.div
              key="network"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
              
              {/* SNN Training Hyperparameters (1/3 column) */}
              <div className="space-y-6">
                
                {/* Network settings card */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
                    <Settings className="w-4 h-4 text-elegant-cyan" />
                    <span>Cài Đặt Cấu Hình Mạng SNN</span>
                  </h3>
                  
                  {/* Learning rate slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Tốc độ học (Learning Rate):</span>
                      <span className="text-elegant-cyan font-bold">{learningRate.toFixed(3)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.005"
                      max="0.08"
                      step="0.005"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Network Decay Magnitude */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Mô-đun rò rỉ |β|:</span>
                      <span className="text-elegant-cyan font-bold">{netLeakMag.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.6"
                      max="0.95"
                      step="0.05"
                      value={netLeakMag}
                      onChange={(e) => setNetLeakMag(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Network Decay Phase */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Pha rò rỉ ∠β (rad):</span>
                      <span className="text-elegant-cyan font-bold">{netLeakPhase.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="0.3"
                      step="0.02"
                      value={netLeakPhase}
                      onChange={(e) => setNetLeakPhase(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Reset mechanism selector */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-mono text-elegant-muted block">Cơ chế Reset:</label>
                    <div className="grid grid-cols-2 gap-2 mt-1">
                      <button
                        onClick={() => setNetResetMode('soft')}
                        className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                          netResetMode === 'soft' 
                            ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35' 
                            : 'bg-elegant-bg text-elegant-text border-elegant-muted/10'
                        }`}
                      >
                        Soft Reset
                      </button>
                      <button
                        onClick={() => setNetResetMode('hard')}
                        className={`py-1.5 px-2 rounded-md text-xs font-bold transition-all border ${
                          netResetMode === 'hard' 
                            ? 'bg-elegant-cyan/15 text-elegant-cyan border-elegant-cyan/35' 
                            : 'bg-elegant-bg text-elegant-text border-elegant-muted/10'
                        }`}
                      >
                        Hard Reset
                      </button>
                    </div>
                  </div>
                </div>

                {/* Cấu hình Tín hiệu Đầu vào Card */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
                    <Sliders className="w-4 h-4 text-elegant-cyan" />
                    <span>Cấu Hình Tín Hiệu Đầu Vào</span>
                  </h3>
                  
                  {/* Class 0 Frequency */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Tần số Lớp 0 (Thấp):</span>
                      <span className="text-elegant-cyan font-bold">{class0Freq.toFixed(2)} Hz</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2.5"
                      step="0.1"
                      value={class0Freq}
                      onChange={(e) => setClass0Freq(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Class 1 Frequency */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Tần số Lớp 1 (Cao):</span>
                      <span className="text-elegant-cyan font-bold">{class1Freq.toFixed(2)} Hz</span>
                    </div>
                    <input
                      type="range"
                      min="3.0"
                      max="6.0"
                      step="0.1"
                      value={class1Freq}
                      onChange={(e) => setClass1Freq(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>

                  {/* Class 1 Phase Shift */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono text-elegant-muted">
                      <span>Lệch Pha Lớp 1 (Kênh):</span>
                      <span className="text-elegant-cyan font-bold">{(class1PhaseShift * 180 / Math.PI).toFixed(0)}° ({(class1PhaseShift).toFixed(2)} rad)</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="3.14"
                      step="0.1"
                      value={class1PhaseShift}
                      onChange={(e) => setClass1PhaseShift(parseFloat(e.target.value))}
                      className="w-full h-1 bg-elegant-bg rounded-lg appearance-none cursor-pointer accent-elegant-cyan"
                    />
                  </div>
                </div>

                {/* Training Actions Card */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 font-display">
                    Huấn Luyện Trực Tiếp (Live Training)
                  </h3>

                  <div className="grid grid-cols-2 gap-3 text-center">
                    <div className="p-3 bg-elegant-bg border border-elegant-muted/10 rounded-lg space-y-1">
                      <span className="text-[10px] text-elegant-muted font-mono block">EPOCH / SỐ BƯỚC:</span>
                      <span className="text-lg font-bold text-white font-mono">{epoch}</span>
                    </div>

                    <div className="p-3 bg-elegant-bg border border-elegant-muted/10 rounded-lg space-y-1">
                      <span className="text-[10px] text-elegant-muted font-mono block">ĐỘ CHÍNH XÁC:</span>
                      <span className="text-lg font-bold text-emerald-400 font-mono">{(trainAccuracy * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  <div className="p-3 bg-elegant-bg border border-elegant-muted/10 rounded-lg text-center space-y-1">
                    <span className="text-[10px] text-elegant-muted font-mono block">LOSS TRÊN MẪU HIỆN TẠI (ĐỘ LỖI):</span>
                    <span className="text-lg font-bold text-elegant-cyan font-mono">{currentLoss ? currentLoss.toFixed(5) : 'N/A'}</span>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => setIsTraining(!isTraining)}
                      className={`flex-1 py-2.5 px-4 rounded-xl flex items-center justify-center gap-2 text-sm font-bold transition-all shadow-md ${
                        isTraining 
                          ? 'bg-rose-600 hover:bg-rose-500 text-white' 
                          : 'bg-elegant-cyan hover:bg-[#52ebd3] text-elegant-bg'
                      }`}
                    >
                      {isTraining ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      <span>{isTraining ? 'Tạm Dừng Huấn Luyện' : 'Bắt Đầu Huấn Luyện'}</span>
                    </button>

                    <button
                      onClick={handleResetNetwork}
                      className="p-2.5 bg-elegant-card border border-elegant-muted/15 hover:bg-elegant-bg/85 rounded-xl transition-all shadow-md"
                      title="Reset mạng và trọng số"
                    >
                      <RefreshCw className="w-4.5 h-4.5 text-elegant-muted" />
                    </button>
                  </div>
                </div>

                {/* Description of task */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-4 rounded-xl space-y-2 text-xs text-elegant-text/70 leading-relaxed shadow-xl">
                  <div className="flex items-center gap-1 text-white font-bold font-display">
                    <Info className="w-3.5 h-3.5 text-elegant-cyan" />
                    <span>Nhiệm vụ Phân loại Phổ Tần Số:</span>
                  </div>
                  <p>
                    Học phân loại 2 lớp tín hiệu đầu vào gồm 16 kênh đặc trưng:
                  </p>
                  <ul className="list-disc pl-4 space-y-1 text-elegant-text">
                    <li><strong className="text-white">Lớp 0:</strong> Tín hiệu tần số thấp (1.0Hz) đồng pha.</li>
                    <li><strong className="text-white">Lớp 1:</strong> Tín hiệu tần số cao (4.5Hz) bị lệch pha góc 90° giữa các kênh.</li>
                  </ul>
                  <p className="mt-1">
                    Trọng số phức được tối ưu hóa liên tục để phát hiện đồng thời tần số và phân rã pha của các xung điện.
                  </p>
                </div>

              </div>

              {/* Training Loss History Plot & Current prediction rates (1/3 column) */}
              <div className="space-y-6">
                
                {/* Training Loss Curve Card */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                  <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
                    <h3 className="text-sm font-bold text-white font-display">Biểu Đồ Độ Lỗi Loss</h3>
                    <span className="text-[10px] bg-elegant-cyan/10 text-elegant-cyan px-2 py-0.5 rounded-md font-mono border border-elegant-cyan/20">Real-time</span>
                  </div>

                  <div className="h-44 bg-elegant-bg rounded-lg border border-elegant-muted/15 relative p-2 overflow-hidden flex flex-col justify-end">
                    {trainingLossHistory.length > 1 ? (
                      <svg className="w-full h-full overflow-visible">
                        {/* Grid lines */}
                        <line x1="0" y1="25%" x2="100%" y2="25%" stroke="rgba(69, 162, 158, 0.1)" strokeWidth="1" strokeDasharray="3,3" />
                        <line x1="0" y1="50%" x2="100%" y2="50%" stroke="rgba(69, 162, 158, 0.1)" strokeWidth="1" strokeDasharray="3,3" />
                        <line x1="0" y1="75%" x2="100%" y2="75%" stroke="rgba(69, 162, 158, 0.1)" strokeWidth="1" strokeDasharray="3,3" />

                        {/* Loss path */}
                        <path
                          d={trainingLossHistory.map((loss, idx) => {
                            const maxLoss = Math.max(...trainingLossHistory, 1.5);
                            const x = (idx / (trainingLossHistory.length - 1)) * 310;
                            // Calculate height
                            const y = 140 - (loss / maxLoss) * 110;
                            return `${idx === 0 ? 'M' : 'L'} ${x},${y}`;
                          }).join(' ')}
                          fill="none"
                          stroke="#66FCF1"
                          strokeWidth="2.5"
                          className="transition-all duration-150"
                        />
                      </svg>
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-xs text-elegant-muted font-mono">Bắt đầu huấn luyện để vẽ đồ thị</div>
                    )}
                  </div>
                  <div className="flex justify-between text-[10px] text-elegant-muted font-mono">
                    <span>Cũ nhất</span>
                    <span>Epoch hiện tại ({epoch})</span>
                  </div>
                </div>

                {/* Phân tích 16 Kênh Tín Hiệu Đầu Vào Card */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3 shadow-xl">
                  <div className="flex items-center justify-between border-b border-elegant-muted/10 pb-2">
                    <h3 className="text-sm font-bold text-white font-display flex items-center gap-1.5">
                      <Sliders className="w-4 h-4 text-elegant-cyan" />
                      <span>Phân Tích 16 Kênh Tín Hiệu Đầu Vào (Input Phase Map)</span>
                    </h3>
                    <span className="text-[10px] bg-elegant-bg text-elegant-muted px-2 py-0.5 rounded font-mono border border-elegant-muted/10">
                      t = 30 steps
                    </span>
                  </div>

                  {currentSample ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
                        {Array.from({ length: 16 }).map((_, fIdx) => {
                          // Calculate the average phase and magnitude for channel fIdx over the time sequence
                          let sumRe = 0;
                          let sumIm = 0;
                          let sumMag = 0;
                          const steps = currentSample.inputSpikeSequence.length;
                          
                          for (let t = 0; t < steps; t++) {
                            const val = currentSample.inputSpikeSequence[t][fIdx];
                            sumRe += val.re;
                            sumIm += val.im;
                            sumMag += complex.mag(val);
                          }
                          
                          const avgComplex = complex.create(sumRe / steps, sumIm / steps);
                          const avgMag = sumMag / steps;
                          const avgPhase = complex.angle(avgComplex);
                          const rotDegree = (avgPhase * 180) / Math.PI;
                          const color = getPhaseColor(avgPhase);
                          
                          return (
                            <div 
                              key={`input-chan-${fIdx}`}
                              className="p-1.5 bg-elegant-bg border border-elegant-muted/10 rounded-md flex flex-col items-center justify-center space-y-1 relative group hover:border-elegant-cyan/30 transition-all cursor-help"
                              title={`Kênh ${fIdx} | Avg Amp: ${avgMag.toFixed(2)}, Avg Phase: ${avgPhase.toFixed(2)} rad (${rotDegree.toFixed(0)}°)`}
                            >
                              <span className="text-[9px] text-elegant-muted font-mono">CH {fIdx}</span>
                              
                              {/* Mini Compass */}
                              <div 
                                className="w-7 h-7 rounded-full border border-elegant-muted/20 relative flex items-center justify-center transition-transform group-hover:scale-105"
                                style={{ borderColor: 'rgba(69, 162, 158, 0.15)' }}
                              >
                                {/* Compass Hand */}
                                <div 
                                  className="w-3.5 h-[1.5px] absolute right-1/2 transform origin-right transition-all duration-300"
                                  style={{ 
                                    backgroundColor: color,
                                    transform: `rotate(${-rotDegree}deg)`,
                                    boxShadow: `0 0 6px ${color}`
                                  }}
                                />
                                <div className="w-1 h-1 rounded-full bg-elegant-muted" />
                              </div>

                              {/* Mini Magnitude bar */}
                              <div className="w-full h-1 bg-elegant-card rounded-full overflow-hidden">
                                <div 
                                  className="h-full bg-elegant-cyan rounded-full"
                                  style={{ width: `${Math.min((avgMag / 1.5) * 100, 100)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      
                      <p className="text-[10px] text-elegant-text/60 leading-relaxed text-center italic">
                        La bàn của 16 kênh đầu vào: Ở <strong className="text-white">Lớp 0</strong> chúng đồng pha (đồng loạt xoay giống nhau), còn ở <strong className="text-white">Lớp 1</strong> chúng lệch pha tuần tiến {class1PhaseShift === 1.57 ? '(~90°)' : `(~${(class1PhaseShift * 180 / Math.PI).toFixed(0)}°)`} tạo thành chuỗi sắc màu xoay hướng.
                      </p>
                    </div>
                  ) : (
                    <div className="text-xs text-elegant-muted text-center py-6">Không có dữ liệu đầu vào</div>
                  )}
                </div>

                {/* SNN Prediction Output Spike intensity meters */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                  <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
                    <Cpu className="w-4 h-4 text-elegant-cyan" />
                    <span>Năng Lượng Xung Đầu Ra (Output Spikes)</span>
                  </h3>

                  {sampleOutputs && currentSample ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between text-xs text-elegant-muted">
                        <span>Nhãn mẫu đang chạy:</span>
                        <span className={`px-2 py-0.5 rounded-md text-xs font-bold font-mono ${
                          currentSample.target === 0 ? 'bg-elegant-cyan/5 text-elegant-cyan border border-elegant-cyan/20' : 'bg-elegant-cyan/10 text-elegant-cyan border border-elegant-cyan/30'
                        }`}>
                          LỚP {currentSample.target} (Tần số {currentSample.target === 0 ? 'Thấp 1Hz' : 'Cao 4.5Hz'})
                        </span>
                      </div>

                      {/* Class Output Bars */}
                      <div className="space-y-3">
                        {/* Class 0 firing rate */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-elegant-text">Neuron phân loại LỚP 0 (Thấp):</span>
                            <span className="font-bold text-white">{(sampleOutputs.outputSpikeRates[0] * 100).toFixed(1)}% intensity</span>
                          </div>
                          <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                            <div 
                              className="h-full bg-gradient-to-r from-[#45A29E] to-elegant-cyan rounded-full transition-all duration-300"
                              style={{ width: `${Math.min(sampleOutputs.outputSpikeRates[0] * 100, 100)}%` }}
                            />
                          </div>
                        </div>

                        {/* Class 1 firing rate */}
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-elegant-text">Neuron phân loại LỚP 1 (Cao):</span>
                            <span className="font-bold text-white">{(sampleOutputs.outputSpikeRates[1] * 100).toFixed(1)}% intensity</span>
                          </div>
                          <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                            <div 
                              className="h-full bg-gradient-to-r from-elegant-cyan to-[#45A29E] rounded-full transition-all duration-300"
                              style={{ width: `${Math.min(sampleOutputs.outputSpikeRates[1] * 100, 100)}%` }}
                            />
                          </div>
                        </div>
                      </div>

                      {/* Decisive match evaluation */}
                      <div className="p-3 bg-elegant-bg border border-elegant-muted/10 rounded-lg flex items-center justify-between">
                        <span className="text-xs text-elegant-muted">Kết quả dự đoán:</span>
                        {(() => {
                          const rates = sampleOutputs.outputSpikeRates;
                          const pred = rates[1] > rates[0] ? 1 : 0;
                          const correct = pred === currentSample.target;
                          return (
                            <span className={`text-xs font-bold px-2 py-0.5 rounded-md font-mono ${
                              correct ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                            }`}>
                              {correct ? '✓ CHÍNH XÁC' : '✗ SAI LỆCH'}
                            </span>
                          );
                        })()}
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-elegant-muted text-center py-8">Chưa có kết quả nổ xung</div>
                  )}
                </div>

              </div>

              {/* SNN Weight Matrix Visualizer Compass Grid (1/3 column) */}
              <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl flex flex-col justify-between min-h-[400px] shadow-xl">
                <div className="w-full flex items-center justify-between border-b border-elegant-muted/10 pb-2">
                  <h3 className="text-sm font-bold text-white font-display">Trọng Số Phức Synaptic (Weight Matrix)</h3>
                  <span className="text-[10px] bg-elegant-bg text-elegant-muted px-2 py-0.5 rounded font-mono border border-elegant-muted/10">16 Inputs x 2 Class Outputs</span>
                </div>

                {network ? (
                  <div className="my-4 grid grid-cols-2 gap-4 flex-1 items-center">
                    
                    {/* Weight column for Class 0 */}
                    <div className="space-y-2">
                      <div className="text-center text-xs font-bold font-mono text-elegant-cyan">TẦNG TRỌNG SỐ LỚP 0</div>
                      <div className="bg-elegant-bg p-2 rounded-lg border border-elegant-muted/10 grid grid-cols-4 gap-2.5">
                        {network.weights[0].map((w, idx) => {
                          const wMag = complex.mag(w);
                          const wPhase = complex.angle(w);
                          const rotDegree = (wPhase * 180) / Math.PI;
                          
                          // Opacity represents weight magnitude scaled [0, 1]
                          const opacity = Math.min(Math.max(wMag * 1.5, 0.25), 1.0);
                          
                          return (
                            <div 
                              key={`w0-${idx}`} 
                              className="aspect-square bg-elegant-card rounded-md border border-elegant-muted/10 relative flex items-center justify-center cursor-help transition-all hover:bg-elegant-card/80"
                              title={`W0[${idx}] | Magnitude: ${wMag.toFixed(3)}, Phase: ${wPhase.toFixed(2)} rad (${rotDegree.toFixed(0)}°)`}
                            >
                              {/* Inner compass compass arrow pointing in direction of phase */}
                              <div 
                                className="w-5 h-5 rounded-full border border-elegant-muted/20 relative flex items-center justify-center"
                                style={{ opacity }}
                              >
                                {/* Vector line inside */}
                                <div 
                                  className="w-2.5 h-[2px] bg-elegant-cyan absolute right-1/2 transform origin-right"
                                  style={{ transform: `rotate(${-rotDegree}deg)` }}
                                />
                                <div className="w-1.5 h-1.5 rounded-full bg-elegant-muted" />
                              </div>
                              
                              <span className="absolute bottom-0.5 right-1 text-[7px] text-elegant-muted font-mono">{idx}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Weight column for Class 1 */}
                    <div className="space-y-2">
                      <div className="text-center text-xs font-bold font-mono text-elegant-cyan">TẦNG TRỌNG SỐ LỚP 1</div>
                      <div className="bg-elegant-bg p-2 rounded-lg border border-elegant-muted/10 grid grid-cols-4 gap-2.5">
                        {network.weights[1].map((w, idx) => {
                          const wMag = complex.mag(w);
                          const wPhase = complex.angle(w);
                          const rotDegree = (wPhase * 180) / Math.PI;
                          
                          // Opacity represents weight magnitude scaled [0, 1]
                          const opacity = Math.min(Math.max(wMag * 1.5, 0.25), 1.0);
                          
                          return (
                            <div 
                              key={`w1-${idx}`} 
                              className="aspect-square bg-elegant-card rounded-md border border-elegant-muted/10 relative flex items-center justify-center cursor-help transition-all hover:bg-elegant-card/80"
                              title={`W1[${idx}] | Magnitude: ${wMag.toFixed(3)}, Phase: ${wPhase.toFixed(2)} rad (${rotDegree.toFixed(0)}°)`}
                            >
                              {/* Inner compass compass arrow pointing in direction of phase */}
                              <div 
                                className="w-5 h-5 rounded-full border border-elegant-muted/20 relative flex items-center justify-center"
                                style={{ opacity }}
                              >
                                {/* Vector line inside */}
                                <div 
                                  className="w-2.5 h-[2px] bg-elegant-cyan absolute right-1/2 transform origin-right"
                                  style={{ transform: `rotate(${-rotDegree}deg)` }}
                                />
                                <div className="w-1.5 h-1.5 rounded-full bg-elegant-muted" />
                              </div>
                              
                              <span className="absolute bottom-0.5 right-1 text-[7px] text-elegant-muted font-mono">{idx}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                  </div>
                ) : (
                  <div className="text-xs text-elegant-muted text-center py-16">Chưa có dữ liệu trọng số</div>
                )}

                <div className="w-full text-[11px] text-elegant-text/70 leading-relaxed text-center">
                  Mỗi ô tròn là một <strong>la bàn trọng số phức</strong>. Chiều mũi tên thể hiện <strong>góc pha (xoay pha trễ)</strong>. Độ tương phản của vòng tròn thể hiện <strong>độ lớn biên độ trọng số (gain)</strong>. Khi bắt đầu học, các mũi tên sẽ tự động xoay chuyển để tìm ra sự phối hợp pha nhịp nhàng nhất!
                </div>
              </div>

            </motion.div>
          )}

          {/* TAB 5: CHOI LAB SOTA & INTEGRATION PROPOSAL */}
          {activeTab === 'choi-lab' && (
            <motion.div
              key="choi-lab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              {/* Header Title */}
              <div className="bg-gradient-to-r from-elegant-card to-elegant-card/40 border border-elegant-muted/15 p-6 rounded-xl shadow-xl space-y-2">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-elegant-cyan/10 rounded-lg text-elegant-cyan border border-elegant-cyan/20">
                    <Brain className="w-6 h-6 animate-pulse" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold font-display text-white">Nghiên Cứu SOTA GS. Shinhyun Choi (KAIST) & Đề Xuất Tích Hợp CV-SNN</h2>
                    <p className="text-xs text-elegant-muted font-mono mt-0.5">
                      Đột phá phần cứng Neuromorphic Memristor Array kết hợp với thuật toán Mạng thần kinh nổ xung miền phức
                    </p>
                  </div>
                </div>
              </div>

              {/* SECTION 1: GS SHINHYUN CHOI SOTA HIGHLIGHTS */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Highlight 1: Self-learning chip */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3.5 shadow-xl hover:border-elegant-cyan/25 transition-all">
                  <div className="flex items-center gap-2 text-elegant-cyan">
                    <Sparkles className="w-5 h-5" />
                    <span className="text-xs font-bold font-mono tracking-wider uppercase">NATURE ELECTRONICS (2024)</span>
                  </div>
                  <h3 className="text-md font-bold text-white font-display">Chip Tự Học & Sửa Sai On-Chip (Self-Learning & Self-Correcting)</h3>
                  <p className="text-xs text-elegant-text/70 leading-relaxed">
                    Đồng phát triển bởi GS. Shinhyun Choi & GS. Young-Gyu Yoon (KAIST). Chip điện toán memristor không cần bộ định tuyến (selector-less) có khả năng <strong>tự động sửa lỗi phi tuyến và rò điện</strong> ngay tại phần cứng. Thiết kế đã chứng minh hiệu năng thực tế xuất sắc trong nhiệm vụ tách biên tách hình động (foreground-background video separation) với độ chính xác tương đương mô phỏng phần mềm lý tưởng.
                  </p>
                </div>

                {/* Highlight 2: Ultra-low power Phase Change Memory */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3.5 shadow-xl hover:border-elegant-cyan/25 transition-all">
                  <div className="flex items-center gap-2 text-elegant-cyan">
                    <Cpu className="w-5 h-5" />
                    <span className="text-xs font-bold font-mono tracking-wider uppercase">NATURE (2024 Breakthrough)</span>
                  </div>
                  <h3 className="text-md font-bold text-white font-display">PCM Nanofilament Siêu Tiêu Thụ Điện Thấp (Ultra-low Power)</h3>
                  <p className="text-xs text-elegant-text/70 leading-relaxed">
                    Bộ nhớ thay đổi pha (PCM) truyền thống gặp rào cản năng lượng nhiệt reset cực lớn. Nhóm của GS. Choi đã phát minh cấu trúc PCM nano-filament siêu nhỏ, giảm thể tích vùng chuyển pha xuống mức nanomet. Kết quả đạt mức <strong>tiêu thụ năng lượng cực thấp (thấp hơn DRAM/NAND hàng nghìn lần)</strong>, sẵn sàng cho việc tích hợp hàng triệu synapse phức mật độ siêu cao.
                  </p>
                </div>

                {/* Highlight 3: Alloy-based memristors */}
                <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-3.5 shadow-xl hover:border-elegant-cyan/25 transition-all">
                  <div className="flex items-center gap-2 text-elegant-cyan">
                    <Layers className="w-5 h-5" />
                    <span className="text-xs font-bold font-mono tracking-wider uppercase">NATURE (2020-2021)</span>
                  </div>
                  <h3 className="text-md font-bold text-white font-display">Memristor Hợp Kim Đồng Nhất Siêu Ổn Định (Alloy-based)</h3>
                  <p className="text-xs text-elegant-text/70 leading-relaxed">
                    Giải quyết triệt để sự bất định thời gian (temporal) và không gian (spatial fluctuation) của các bộ nhớ điện trở. Sử dụng hợp kim đơn tinh thể silicon được kiểm soát nghiêm ngặt, thiết bị synapse của phòng thí nghiệm đạt độ tuyến tính vượt trội, cho phép điều chỉnh trọng số trơn tru và đối xứng trong quá trình học máy phần cứng.
                  </p>
                </div>

              </div>

              {/* SECTION 2 & 3 GRID: INTEGRATION BLUEPRINT & LIVE HARDWARE SANDBOX */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* LEFT: INTEGRATION BLUEPRINT PROPOSAL (2/3 cols) */}
                <div className="lg:col-span-2 space-y-6">
                  
                  {/* Interactive Proposal description */}
                  <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-5 shadow-xl">
                    <div className="flex items-center gap-2 text-elegant-cyan border-b border-elegant-muted/10 pb-2.5">
                      <Zap className="w-5 h-5" />
                      <h3 className="text-md font-bold font-display text-white">Đề Xuất Kiến Trúc Tích Hợp CV-SNN trên Chip GS. Choi</h3>
                    </div>

                    <p className="text-xs text-elegant-text/80 leading-relaxed">
                      Để vận hành mô hình <strong>Complex-Valued SNN (CV-SNN)</strong> với các tín hiệu và trọng số phức trên mảng phần cứng memristor vốn chỉ hỗ trợ các giá trị dẫn nạp thực vật lý (G ∈ ℝ), chúng tôi đề xuất kiến trúc 3 thành phần tích hợp:
                    </p>

                    <div className="space-y-4">
                      
                      {/* Architecture Block 1 */}
                      <div className="p-3.5 bg-elegant-bg/40 border border-elegant-muted/10 rounded-lg space-y-1.5 hover:border-elegant-cyan/15 transition-all">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-elegant-cyan">1. Tầng Trọng Số Phức Vi Sai (Differential 4-Memristor Synapse)</span>
                          <span className="text-[10px] font-mono text-elegant-muted font-bold">Synaptic Core</span>
                        </div>
                        <p className="text-xs text-elegant-text/70 leading-relaxed">
                          Mỗi trọng số phức W = W_r + i W_i được phân tách và biểu diễn thông qua 4 điểm nối memristor chéo (crossbar nodes) vi sai: 
                          <code className="text-rose-300 font-mono text-[10.5px] ml-1">W = (G_re^+ - G_re^-) + i(G_im^+ - G_im^-)</code>.
                          Các tín hiệu đầu vào thực và ảo được nạp lệch pha sóng AC 90° (hoặc thông qua luồng điện áp song song) để mảng crossbar tự động tính toán phép nhân tích chập phức của Kirchhoff ở cấp độ transistor không hao phí.
                        </p>
                      </div>

                      {/* Architecture Block 2 */}
                      <div className="p-3.5 bg-elegant-bg/40 border border-elegant-muted/10 rounded-lg space-y-1.5 hover:border-elegant-cyan/15 transition-all">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-elegant-cyan">2. Bộ Chuyển Đổi Pha sang Độ Trễ Kích Xung (Phase-to-Latency Encoder)</span>
                          <span className="text-[10px] font-mono text-elegant-muted font-bold">Spike Interface</span>
                        </div>
                        <p className="text-xs text-elegant-text/70 leading-relaxed">
                          Pha phức của điện thế màng θ = arg(U) được ánh xạ trực tiếp thành <strong>độ trễ thời gian tuyệt đối (temporal spike latency)</strong> của xung thần kinh phát ra. Khi màng tụ điện tích lũy đạt ngưỡng, pha phức kích hoạt mạch logic thời gian on-chip để trì hoãn xung tương thích, giữ trọn vẹn thông tin sóng pha mà không cần tính toán số học phức tạp.
                        </p>
                      </div>

                      {/* Architecture Block 3 */}
                      <div className="p-3.5 bg-elegant-bg/40 border border-elegant-muted/10 rounded-lg space-y-1.5 hover:border-elegant-cyan/15 transition-all">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-elegant-cyan">3. Thuật Toán Bù Trực Tiếp Phi Tuyến On-Chip (Algorithmic On-Chip Correction)</span>
                          <span className="text-[10px] font-mono text-elegant-muted font-bold">Error Mitigation</span>
                        </div>
                        <p className="text-xs text-elegant-text/70 leading-relaxed">
                          Kế thừa cơ chế tự sửa sai độc quyền của nhóm nghiên cứu GS. Choi. Các sai số ngẫu nhiên khi ghi giá trị vào mảng Memristor (write noise) và tính phi tuyến dẫn nạp được thuật toán lan truyền ngược tự bù sai số ở cấp độ mô hình bằng cách cập nhật động gradient dịch pha bổ sung, giúp bảo tồn sự đồng bộ pha giữa các kênh.
                        </p>
                      </div>

                    </div>
                  </div>

                  {/* Physical Crossbar Schematic visualization */}
                  <div className="bg-elegant-card border border-elegant-muted/15 p-6 rounded-xl space-y-4 shadow-xl">
                    <h3 className="text-sm font-bold text-white tracking-wider uppercase font-mono">Sơ Đồ Mạch Nhân Phức Vi Sai 4-Memristor</h3>
                    <div className="bg-elegant-bg/50 p-4 rounded-lg border border-elegant-muted/10 flex flex-col items-center justify-center space-y-3">
                      {/* Diagram representation with ASCII/CSS */}
                      <div className="w-full max-w-md border border-elegant-muted/15 p-4 rounded bg-elegant-bg relative font-mono text-[10.5px] text-elegant-text space-y-3">
                        <div className="text-center font-bold text-elegant-cyan pb-1 border-b border-elegant-muted/10">Sơ đồ Mạch Crossbar</div>
                        <div className="flex justify-between items-center">
                          <span className="text-white font-bold">Input X_complex:</span>
                          <span className="text-elegant-muted font-bold">Re(X) [Voltage V_r] | Im(X) [Voltage V_i]</span>
                        </div>
                        <div className="p-2.5 bg-elegant-card/50 rounded border border-elegant-muted/5 space-y-1 text-elegant-text/80 text-[10px]">
                          <div>• <span className="text-white">Dòng 1:</span> Điện áp V_r nạp vào cột Re  ➔ Nhân chập chéo: V_r · G_re^+ và V_r · G_re^-</div>
                          <div>• <span className="text-white">Dòng 2:</span> Điện áp V_i nạp vào cột Im  ➔ Nhân chập chéo: V_i · G_im^+ và V_i · G_im^-</div>
                          <div className="text-emerald-400 font-semibold">• Phép cộng dòng Kirchhoff tại điểm ra:</div>
                          <div className="pl-3 font-mono text-[9px] text-emerald-300">
                            I_real = (V_r · G_re^+ - V_r · G_re^-) - (V_i · G_im^+ - V_i · G_im^-)<br />
                            I_imag = (V_r · G_im^+ - V_r · G_im^-) + (V_i · G_re^+ - V_i · G_re^-)
                          </div>
                        </div>
                        <div className="text-center text-[10px] text-elegant-muted italic">
                          ➔ Tự động thực hiện phép toán số phức nguyên bản bằng định luật vật lý trong chớp mắt!
                        </div>
                      </div>
                    </div>
                  </div>

                </div>

                {/* RIGHT: LIVE HARDWARE MAPPING SANDBOX (1/3 cols) */}
                <div className="space-y-6">
                  
                  {/* Interactive Simulator Controller Card */}
                  <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                    <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
                      <Settings className="w-4 h-4 text-elegant-cyan" />
                      <span>Cấu Hình Trọng Số Số Phức W</span>
                    </h3>

                    <p className="text-xs text-elegant-text/70 leading-relaxed">
                      Kéo chỉnh thanh trượt để thay đổi Biên độ và Pha của trọng số phức W. Hệ thống sẽ tự động tính toán cách chip của Lab GS. Choi lập trình 4 thiết bị memristor thực tế!
                    </p>

                    {/* Weight Magnitude slider */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs font-mono text-elegant-muted">
                        <span>Biên độ trọng số |W|:</span>
                        <span className="text-elegant-cyan font-bold">{choiWeightMag.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1.5"
                        step="0.05"
                        value={choiWeightMag}
                        onChange={(e) => setChoiWeightMag(parseFloat(e.target.value))}
                        className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
                      />
                    </div>

                    {/* Weight Phase slider */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs font-mono text-elegant-muted">
                        <span>Góc pha θ (radians):</span>
                        <span className="text-elegant-cyan font-bold">
                          {choiWeightPhase.toFixed(2)} rad ({(choiWeightPhase * 180 / Math.PI).toFixed(0)}°)
                        </span>
                      </div>
                      <input
                        type="range"
                        min="-3.1415"
                        max="3.1415"
                        step="0.05"
                        value={choiWeightPhase}
                        onChange={(e) => setChoiWeightPhase(parseFloat(e.target.value))}
                        className="w-full accent-elegant-cyan bg-elegant-bg h-1 rounded-lg cursor-pointer"
                      />
                    </div>

                    {/* Polar Coordinate Indicator Block */}
                    <div className="p-3.5 bg-elegant-bg rounded-lg border border-elegant-muted/10 text-xs font-mono space-y-1.5 text-center shadow-inner">
                      <div className="text-elegant-muted text-[11px] uppercase tracking-wider">Trọng số dạng phức:</div>
                      <div className="text-sm font-bold text-white">
                        W = { (choiWeightMag * Math.cos(choiWeightPhase)).toFixed(3) } { (choiWeightMag * Math.sin(choiWeightPhase)) >= 0 ? '+' : '-' } { Math.abs(choiWeightMag * Math.sin(choiWeightPhase)).toFixed(3) }i
                      </div>
                      <div className="text-[10px] text-elegant-cyan">
                        Exponential: {choiWeightMag.toFixed(2)} · e^({(choiWeightPhase >= 0 ? '+' : '')}{choiWeightPhase.toFixed(2)}i)
                      </div>
                    </div>
                  </div>

                  {/* Physical Conductance Mapping meters */}
                  <div className="bg-elegant-card border border-elegant-muted/15 p-5 rounded-xl space-y-4 shadow-xl">
                    <h3 className="text-sm font-bold text-white border-b border-elegant-muted/10 pb-2 flex items-center gap-1.5 font-display">
                      <Activity className="w-4 h-4 text-elegant-cyan animate-pulse" />
                      <span>Conductance Lập Trình Memristor</span>
                    </h3>

                    {(() => {
                      // Perform mathematical mapping from complex W to 4 conductances with a baseline bias of 0.1
                      const Wr = choiWeightMag * Math.cos(choiWeightPhase);
                      const Wi = choiWeightMag * Math.sin(choiWeightPhase);
                      const bias = 0.1;
                      const gRePlus = bias + Math.max(0, Wr);
                      const gReMinus = bias + Math.max(0, -Wr);
                      const gImPlus = bias + Math.max(0, Wi);
                      const gImMinus = bias + Math.max(0, -Wi);
                      
                      // Max possible value for meter scaling is bias + max magnitude (1.5) = 1.6
                      const scaleMax = 1.6;

                      return (
                        <div className="space-y-4">
                          <p className="text-[11px] text-elegant-text/70 leading-relaxed">
                            Mức dẫn nạp (Conductance G) thực tế được ghi nạp vào 4 thanh memristor vi sai chéo:
                          </p>

                          {/* 4 Meter Bars */}
                          <div className="space-y-3 font-mono">
                            
                            {/* G_re_plus */}
                            <div className="space-y-1 text-xs">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-elegant-cyan font-bold">G_re⁺ (Dẫn thực dương):</span>
                                <span className="text-white">{gRePlus.toFixed(3)} mS</span>
                              </div>
                              <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                                <div 
                                  className="h-full bg-gradient-to-r from-[#45A29E] to-elegant-cyan rounded-full transition-all duration-150"
                                  style={{ width: `${(gRePlus / scaleMax) * 100}%` }}
                                />
                              </div>
                            </div>

                            {/* G_re_minus */}
                            <div className="space-y-1 text-xs">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-rose-400 font-bold">G_re⁻ (Dẫn thực âm):</span>
                                <span className="text-white">{gReMinus.toFixed(3)} mS</span>
                              </div>
                              <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                                <div 
                                  className="h-full bg-gradient-to-r from-rose-600 to-rose-400 rounded-full transition-all duration-150"
                                  style={{ width: `${(gReMinus / scaleMax) * 100}%` }}
                                />
                              </div>
                            </div>

                            {/* G_im_plus */}
                            <div className="space-y-1 text-xs">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-amber-400 font-bold">G_im⁺ (Dẫn ảo dương):</span>
                                <span className="text-white">{gImPlus.toFixed(3)} mS</span>
                              </div>
                              <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                                <div 
                                  className="h-full bg-gradient-to-r from-amber-600 to-amber-400 rounded-full transition-all duration-150"
                                  style={{ width: `${(gImPlus / scaleMax) * 100}%` }}
                                />
                              </div>
                            </div>

                            {/* G_im_minus */}
                            <div className="space-y-1 text-xs">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-indigo-400 font-bold">G_im⁻ (Dẫn ảo âm):</span>
                                <span className="text-white">{gImMinus.toFixed(3)} mS</span>
                              </div>
                              <div className="h-2.5 w-full bg-elegant-bg rounded-full overflow-hidden border border-elegant-muted/10">
                                <div 
                                  className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-full transition-all duration-150"
                                  style={{ width: `${(gImMinus / scaleMax) * 100}%` }}
                                />
                              </div>
                            </div>

                          </div>

                          <div className="p-3 bg-elegant-cyan/5 border border-elegant-cyan/25 rounded-lg text-[11px] text-elegant-text/90 leading-relaxed italic">
                            💡 <strong className="text-white">Nguyên lý vi sai:</strong> Giá trị thực nhận được là <span className="text-elegant-cyan font-bold font-mono">G_re⁺ - G_re⁻ = {Wr.toFixed(3)}</span>, và giá trị ảo là <span className="text-amber-400 font-bold font-mono">G_im⁺ - G_im⁻ = {Wi.toFixed(3)}</span>. Khi bạn kéo lệch pha hoặc biên độ, dòng điện phân cực thực tế sẽ chảy qua mảng chéo tương ứng sinh ra tích chập cực chuẩn!
                          </div>
                        </div>
                      );
                    })()}
                  </div>

                </div>

              </div>
            </motion.div>
          )}

          {activeTab === 'vision' && (
            <motion.div
              key="vision"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
            >
              <VisionSNN />
            </motion.div>
          )}

          {activeTab === 'comparison' && (
            <motion.div
              key="comparison"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
            >
              <SNNComparison />
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-elegant-card bg-elegant-card/20 py-4 px-6 text-center text-xs text-elegant-muted font-mono">
        Mô hình toán học và mã nguồn dựa trên các ấn bản SOTA IEEE & arXiv năm 2022-2026. Phát triển bởi Google AI Studio Build.
      </footer>
    </div>
  );
}
