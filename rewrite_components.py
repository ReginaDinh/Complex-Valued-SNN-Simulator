import re
import os

with open('src/App.tsx', 'r') as f:
    app_code = f.read()

def extract_section(start_marker, end_marker, text):
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    return text[start:end]

# Extract States
choi_state = extract_section("// --- SHINHYUN CHOI LAB INTEGRATION STATE ---", "// --- TAB 3: NEURON SIMULATION STATE ---", app_code)
neuron_state = extract_section("// --- TAB 3: NEURON SIMULATION STATE ---", "// --- TAB 4: SNN NETWORK TRAINING STATE ---", app_code)
network_state = extract_section("// --- TAB 4: SNN NETWORK TRAINING STATE ---", "// Copy code utility", app_code)

# Additional helper functions
helper_funcs = extract_section("// Copy code utility", "return (", app_code)
# helper_funcs contains handleCopyCode, getPhaseColor, etc.
get_phase_color = """
  const getPhaseColor = (radPhase: number) => {
    const deg = ((radPhase + Math.PI) / (2 * Math.PI)) * 360;
    return `hsl(${deg.toFixed(1)}, 90%, 60%)`;
  };
"""

# Extract JSX
def get_jsx(tab_id):
    pattern = rf"\{{activeTab === '{tab_id}' && \(\s*(<motion\.div.*?</motion\.div>)\s*\)\}}"
    match = re.search(pattern, app_code, re.DOTALL)
    if match:
        return match.group(1)
    return ""

theory_jsx = get_jsx("theory")
pytorch_jsx = get_jsx("pytorch")
neuron_jsx = get_jsx("neuron")
network_jsx = get_jsx("network")
choi_jsx = get_jsx("choi-lab")

# Imports
common_imports = """import { useState, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { 
  Brain, Cpu, Layers, Activity, Play, Pause, RefreshCw, 
  Copy, Check, Settings, Info, Sliders, Zap, TrendingUp, Sparkles, BookOpen, Terminal,
  ThumbsUp, AlertTriangle, Eye, GitCompare
} from 'lucide-react';
import { 
  complex, Complex, ComplexLIFNeuron, TSComplexSNN, 
  generateSpectralSignal, SignalSample 
} from '../simulationEngine';
"""

with open('src/components/TheoryTab.tsx', 'w') as f:
    f.write(common_imports + "\nexport default function TheoryTab() {\n  return (\n" + theory_jsx + "\n  );\n}\n")

with open('src/components/PytorchTab.tsx', 'w') as f:
    f.write(common_imports + "import { pytorchCodeString } from '../pytorchCode';\n" + "\nexport default function PytorchTab() {\n" + """
  const [copied, setCopied] = useState(false);
  const handleCopyCode = () => {
    navigator.clipboard.writeText(pytorchCodeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
""" + "  return (\n" + pytorch_jsx + "\n  );\n}\n")

# For Neuron Tab, we need the canvas rendering logic which uses activeTab
# Since it's now isolated, we don't need activeTab check
neuron_state_fixed = neuron_state.replace("|| activeTab !== 'neuron'", "").replace(", activeTab", "")

with open('src/components/NeuronTab.tsx', 'w') as f:
    f.write(common_imports + "\nexport default function NeuronTab() {\n" + neuron_state_fixed + "  return (\n" + neuron_jsx + "\n  );\n}\n")

network_state_fixed = network_state.replace("|| activeTab !== 'network'", "").replace(", activeTab", "")

with open('src/components/NetworkTab.tsx', 'w') as f:
    f.write(common_imports + "\nexport default function NetworkTab() {\n" + network_state_fixed + get_phase_color + "  return (\n" + network_jsx + "\n  );\n}\n")

with open('src/components/ChoiLabTab.tsx', 'w') as f:
    f.write(common_imports + "\nexport default function ChoiLabTab() {\n" + choi_state + "  return (\n" + choi_jsx + "\n  );\n}\n")

# Now rewrite App.tsx to just render the tabs
app_skeleton = """import { useState } from 'react';
import { 
  Brain, BookOpen, GitCompare, Terminal, Activity, Layers, Eye, Cpu
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import TheoryTab from './components/TheoryTab';
import SNNComparison from './components/SNNComparison';
import PytorchTab from './components/PytorchTab';
import NeuronTab from './components/NeuronTab';
import NetworkTab from './components/NetworkTab';
import VisionSNN from './components/VisionSNN';
import ChoiLabTab from './components/ChoiLabTab';

export default function App() {
  const [activeTab, setActiveTab] = useState<'theory' | 'comparison' | 'pytorch' | 'neuron' | 'network' | 'choi-lab' | 'vision'>('theory');

  return (
    <div className="min-h-screen bg-elegant-bg text-elegant-text selection:bg-elegant-cyan/30 flex flex-col font-sans overflow-x-hidden">
      {/* HEADER */}
      <header className="border-b border-elegant-card bg-elegant-card/50 backdrop-blur-md sticky top-0 z-50 shadow-xl">
        <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-elegant-cyan blur-md opacity-30 rounded-full animate-pulse-slow"></div>
              <div className="bg-gradient-to-tr from-elegant-cyan to-indigo-400 p-2 rounded-xl relative shadow-lg">
                <Brain className="w-7 h-7 text-elegant-bg" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight text-white font-display">Complex-Valued SNN</h1>
              <p className="text-xs text-elegant-cyan font-bold tracking-widest font-mono uppercase mt-0.5">Mạng Nơ-ron Xung Số Phức</p>
            </div>
          </div>
          
          <div className="flex bg-elegant-bg/80 p-1.5 rounded-lg border border-elegant-muted/20 shadow-inner overflow-x-auto max-w-full hide-scrollbar snap-x">
            {[
              { id: 'theory', icon: BookOpen, label: 'Lý Thuyết' },
              { id: 'comparison', icon: GitCompare, label: 'So Sánh SNN' },
              { id: 'pytorch', icon: Terminal, label: 'PyTorch Code' },
              { id: 'neuron', icon: Activity, label: 'Neuron Sim' },
              { id: 'network', icon: Layers, label: 'Network Train' },
              { id: 'vision', icon: Eye, label: 'Vision SNN' },
              { id: 'choi-lab', icon: Cpu, label: 'Choi Lab HW' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-md text-xs font-bold transition-all whitespace-nowrap snap-center font-mono ${
                  activeTab === tab.id 
                    ? 'bg-elegant-card text-elegant-cyan shadow-md' 
                    : 'text-elegant-muted hover:text-elegant-text hover:bg-elegant-card/50'
                }`}
              >
                <tab.icon className={`w-4 h-4 ${activeTab === tab.id ? 'text-elegant-cyan' : 'opacity-70'}`} />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      <main className="flex-grow max-w-7xl mx-auto w-full px-6 py-8 relative">
        <AnimatePresence mode="wait">
          {activeTab === 'theory' && <TheoryTab />}
          {activeTab === 'comparison' && (
            <motion.div key="comparison" initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -15 }} transition={{ duration: 0.2 }}>
              <SNNComparison />
            </motion.div>
          )}
          {activeTab === 'pytorch' && <PytorchTab />}
          {activeTab === 'neuron' && <NeuronTab />}
          {activeTab === 'network' && <NetworkTab />}
          {activeTab === 'vision' && (
            <motion.div key="vision" initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -15 }} transition={{ duration: 0.2 }}>
              <VisionSNN />
            </motion.div>
          )}
          {activeTab === 'choi-lab' && <ChoiLabTab />}
        </AnimatePresence>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-elegant-card bg-elegant-card/20 py-4 px-6 text-center text-xs text-elegant-muted font-mono">
        Mô hình toán học và mã nguồn dựa trên các ấn bản SOTA IEEE & arXiv năm 2022-2026. Phát triển bởi Google AI Studio Build.
      </footer>
    </div>
  );
}
"""
with open('src/App.tsx', 'w') as f:
    f.write(app_skeleton)

print("Extraction completed!")
