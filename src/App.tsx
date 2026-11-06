import { useState, useEffect, lazy, Suspense } from 'react';
import { useLanguage } from './contexts/LanguageContext';
import {
  BookOpen,
  Terminal,
  Activity,
  Layers,
  Cpu,
  Sun,
  Moon,
  GitCompare,
  Eye,
  Menu,
  X,
  Sparkles,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

import TheoryTab from './components/TheoryTab';
// Non-default tabs are loaded on first visit to keep the initial bundle small
const SNNComparison = lazy(() => import('./components/SNNComparison'));
const PytorchTab = lazy(() => import('./components/PytorchTab'));
const NeuronTab = lazy(() => import('./components/NeuronTab'));
const NetworkTab = lazy(() => import('./components/NetworkTab'));
const VisionSNN = lazy(() => import('./components/VisionSNN'));
const NeuromorphicHardwareLab = lazy(() => import('./components/NeuromorphicHardwareLab'));
import PerformanceMonitor, { ActiveModelTabId } from './components/PerformanceMonitor';

export type NavTabId =
  | 'theory'
  | 'comparison'
  | 'neuron'
  | 'network'
  | 'vision'
  | 'hardware'
  | 'pytorch';

interface NavGroup {
  groupEn: string;
  groupVi: string;
  tabs: {
    id: NavTabId;
    index: string;
    icon: typeof BookOpen;
    labelEn: string;
    labelVi: string;
    subtitleEn: string;
    subtitleVi: string;
    domainMetric: string;
  }[];
}

function TabFallback({ isEn }: { isEn: boolean }) {
  return (
    <div className="py-16 text-center text-xs font-mono text-elegant-muted">
      {isEn ? 'Loading module…' : 'Đang tải phân hệ…'}
    </div>
  );
}

export default function App() {
  const { language, setLanguage } = useLanguage();
  const isEn = language === 'en';

  const [activeTab, setActiveTab] = useState<NavTabId>('theory');
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState<boolean>(false);

  const [themeMode, setThemeMode] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('app_theme_mode');
    return saved === 'light' ? 'light' : 'dark';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (themeMode === 'light') {
      root.classList.add('theme-light');
    } else {
      root.classList.remove('theme-light');
    }
    localStorage.setItem('app_theme_mode', themeMode);
  }, [themeMode]);

  const navGroups: NavGroup[] = [
    {
      groupEn: 'FOUNDATIONS',
      groupVi: 'LÝ THUYẾT & ĐỐI CHIẾU',
      tabs: [
        {
          id: 'theory',
          index: '01',
          icon: BookOpen,
          labelEn: 'Phasor Synaptic Algebra',
          labelVi: 'Lý Thuyết & Đại Số Phasor',
          subtitleEn:
            'Complex phasor synaptic transmission, Karatsuba 3-MAC reduction, and Wirtinger gradient calculus',
          subtitleVi:
            'Đại số synapse phasor, giảm độ phức tạp Karatsuba 3M và giải tích đạo hàm Wirtinger miền phức',
          domainMetric: 'Domain: U(t) ∈ ℂ · Karatsuba 3-MAC',
        },
        {
          id: 'comparison',
          index: '02',
          icon: GitCompare,
          labelEn: 'SNN vs cLIF Dynamics',
          labelVi: 'So Sánh LIF vs cLIF',
          subtitleEn:
            'Quantitative comparative dynamics between 1D scalar rate-coded LIF and 2D phase-locked cLIF',
          subtitleVi:
            'Đối chiếu định lượng giữa mạng LIF 1D mã hóa tần số và mạng cLIF 2D khóa pha thời gian thực',
          domainMetric: '1D Scalar ℝ vs 2D Phasor ℂ',
        },
      ],
    },
    {
      groupEn: 'SIMULATION LABS',
      groupVi: 'MÔ PHỎNG THỰC NGHIỆM',
      tabs: [
        {
          id: 'neuron',
          index: '03',
          icon: Activity,
          labelEn: 'cLIF Neuron & Bode',
          labelVi: 'Neuron cLIF & Phổ Bode',
          subtitleEn:
            'Single cLIF state-space spiral trajectory, circular thresholding, and subthreshold frequency resonance spectrum',
          subtitleVi:
            'Quỹ đạo không gian trạng thái xoắn ốc 2D, reset vòng bảo toàn pha và phổ cộng hưởng tần số Bode',
          domainMetric: 'f₀ = ∠β / 2π · Pre-Reset Readout',
        },
        {
          id: 'network',
          index: '04',
          icon: Layers,
          labelEn: '16-Ch Spectral Classifier',
          labelVi: 'Mạng Phổ 16-Kênh',
          subtitleEn:
            '16-channel carrier demodulation, Wirtinger BPTT training, automated noise stress test, and confusion matrix',
          subtitleVi:
            'Phân loại phổ 16 kênh với giải mã sóng mang, lan truyền ngược Wirtinger BPTT và kiểm định chống nhiễu',
          domainMetric: '16×2 Complex SNN · T = 30 Steps',
        },
        {
          id: 'vision',
          index: '05',
          icon: Eye,
          labelEn: '8×8 Spatial Phase Vision',
          labelVi: 'Thị Giác Pha 8×8',
          subtitleEn:
            'Spatial phase encoding on 8×8 patterns, conjugate matched-filter crossbar, and 4-class empirical benchmark',
          subtitleVi:
            'Mã hóa sóng pha không gian 8×8, bộ lọc phối hợp liên hợp phức và kiểm định thực nghiệm 4 lớp',
          domainMetric: '64×4 Conjugate Crossbar · G_max',
        },
      ],
    },
    {
      groupEn: 'HARDWARE & CODE',
      groupVi: 'VI MẠCH & MÃ NGUỒN',
      tabs: [
        {
          id: 'hardware',
          index: '06',
          icon: Cpu,
          labelEn: 'Neuromorphic Hardware',
          labelVi: 'Phần Cứng Memristor',
          subtitleEn:
            'KAIST SOTA research, physical crossbar schematic, live Kirchhoff complex multiplier, and PyVISA SCPI driver',
          subtitleVi:
            'Đột phá SOTA KAIST, sơ đồ kiến trúc vi mạch, kiểm chứng Kirchhoff thực tế và driver điều khiển PyVISA',
          domainMetric: '4-Quadrant Differential PCM · KAIST',
        },
        {
          id: 'pytorch',
          index: '07',
          icon: Terminal,
          labelEn: 'PyTorch SDK Reference',
          labelVi: 'Mã Nguồn PyTorch SDK',
          subtitleEn:
            'Production-grade torch.complex64 autograd modules: KaratsubaComplexLinear and VonMisesSurrogate',
          subtitleVi:
            'Mã nguồn PyTorch chuẩn hóa với KaratsubaComplexLinear, VonMisesSurrogate và PhaseAttention',
          domainMetric: 'torch.complex64 · Autograd Graph',
        },
      ],
    },
  ];

  const allTabs = navGroups.flatMap(g => g.tabs);
  const currentTabInfo = allTabs.find(t => t.id === activeTab) || allTabs[0];
  const currentGroupInfo = navGroups.find(g => g.tabs.some(t => t.id === activeTab));

  const handleSelectTab = (tabId: NavTabId) => {
    setActiveTab(tabId);
    setMobileSidebarOpen(false);
  };

  return (
    <div className="min-h-screen bg-elegant-bg text-elegant-text selection:bg-elegant-cyan/25 flex font-sans overflow-x-hidden">
      {/* MOBILE BACKDROP OVERLAY */}
      {mobileSidebarOpen && (
        <div
          onClick={() => setMobileSidebarOpen(false)}
          className="fixed inset-0 bg-black/60 backdrop-blur-xs z-40 lg:hidden"
        />
      )}

      {/* LEFT NAVIGATION SIDEBAR (STICKY ON DESKTOP, DRAWER ON MOBILE) */}
      <aside
        className={`fixed lg:sticky top-0 left-0 z-50 h-screen w-64 sm:w-72 bg-elegant-card border-r border-slate-800/80 flex flex-col justify-between transition-transform duration-200 ease-in-out shrink-0 select-none ${
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Sidebar Header / Brand */}
        <div className="p-4 border-b border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-elegant-cyan/15 border border-elegant-cyan/30 text-elegant-cyan">
                <Sparkles className="w-4 h-4" />
              </div>
              <span className="font-bold font-display text-white text-sm sm:text-base tracking-tight">
                Complex-Valued SNN
              </span>
            </div>
            <button
              onClick={() => setMobileSidebarOpen(false)}
              className="p-1 text-elegant-muted hover:text-white rounded-md lg:hidden cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="flex items-center justify-between text-[10px] font-mono text-elegant-muted">
            <span className="text-elegant-cyan font-semibold">Neuromorphic Workbench</span>
            <span className="text-slate-500">v2.5</span>
          </div>
        </div>

        {/* Sidebar Navigation Items List */}
        <div className="flex-1 overflow-y-auto py-3 px-2 space-y-4 hide-scrollbar">
          {navGroups.map((group, gIdx) => (
            <div key={gIdx} className="space-y-1">
              <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 px-3 py-1">
                {isEn ? group.groupEn : group.groupVi}
              </div>
              {group.tabs.map(tab => {
                const isActive = activeTab === tab.id;
                const IconComp = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => handleSelectTab(tab.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer group text-left ${
                      isActive
                        ? 'bg-elegant-cyan/15 text-white font-semibold border border-elegant-cyan/30 shadow-xs'
                        : 'text-elegant-muted hover:text-white hover:bg-elegant-bg/80 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <IconComp
                        className={`w-4 h-4 shrink-0 transition-colors ${
                          isActive
                            ? 'text-elegant-cyan'
                            : 'text-slate-400 group-hover:text-white'
                        }`}
                      />
                      <span className="truncate">
                        {isEn ? tab.labelEn : tab.labelVi}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {/* Sidebar Footer Controls: Theme Toggle & Language Switcher */}
        <div className="p-3.5 border-t border-slate-800/80 bg-elegant-bg/50 space-y-2.5">
          <div className="flex items-center justify-between gap-2">
            {/* Language Switcher Buttons */}
            <div className="flex items-center p-0.5 bg-elegant-card rounded-md border border-slate-800/80 text-xs font-mono">
              <button
                onClick={() => setLanguage('vi')}
                className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                  language === 'vi'
                    ? 'bg-elegant-cyan/15 text-elegant-cyan font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                Tiếng Việt
              </button>
              <button
                onClick={() => setLanguage('en')}
                className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
                  language === 'en'
                    ? 'bg-elegant-cyan/15 text-elegant-cyan font-bold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                English
              </button>
            </div>

            {/* Theme Toggle Button */}
            <button
              onClick={() => setThemeMode(themeMode === 'dark' ? 'light' : 'dark')}
              className="p-1.5 bg-elegant-card border border-slate-800/80 hover:border-elegant-cyan/50 rounded-md text-elegant-muted hover:text-white transition-colors cursor-pointer"
              title={
                isEn
                  ? 'Toggle Obsidian Dark / IEEE Paper Light Theme'
                  : 'Chuyển đổi giao diện Tối / Sáng'
              }
            >
              {themeMode === 'dark' ? (
                <Sun className="w-4 h-4 text-amber-400" />
              ) : (
                <Moon className="w-4 h-4 text-elegant-cyan" />
              )}
            </button>
          </div>

          <div className="text-[10px] font-mono text-slate-500 text-center truncate">
            {currentTabInfo.domainMetric}
          </div>
        </div>
      </aside>

      {/* RIGHT MAIN WORKBENCH CONTENT AREA */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen">
        {/* Top Context Bar */}
        <header className="border-b border-slate-800/80 bg-elegant-card/70 backdrop-blur-md sticky top-0 z-30">
          <div className="px-4 sm:px-8 h-14 flex items-center justify-between gap-4">
            {/* Mobile Sidebar Hamburger Toggle & Active Tab Title */}
            <div className="flex items-center gap-3 min-w-0">
              <button
                onClick={() => setMobileSidebarOpen(true)}
                className="p-1.5 rounded-lg bg-elegant-bg border border-slate-800/80 text-white lg:hidden cursor-pointer"
                title="Open Navigation"
              >
                <Menu className="w-5 h-5" />
              </button>

              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[11px] font-mono text-elegant-muted">
                  <span className="text-elegant-cyan font-bold uppercase tracking-wider">
                    {currentGroupInfo ? (isEn ? currentGroupInfo.groupEn : currentGroupInfo.groupVi) : 'CV-SNN'}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="text-slate-400 truncate">
                    {currentTabInfo.domainMetric}
                  </span>
                </div>
                <h1 className="text-sm sm:text-base font-bold font-display text-white truncate">
                  {isEn ? currentTabInfo.labelEn : currentTabInfo.labelVi}
                </h1>
              </div>
            </div>

            {/* Quick Status / Mode Indicator on Right */}
            <div className="hidden md:flex items-center gap-2 text-xs font-mono text-slate-400 shrink-0">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>{isEn ? 'Simulation Mode' : 'Chế độ mô phỏng'}</span>
            </div>
          </div>
        </header>

        {/* Tab Subtitle & Context Ribbon */}
        <section className="border-b border-slate-800/60 bg-elegant-card/30 px-4 sm:px-8 py-2.5">
          <p className="text-xs text-elegant-text/75 font-mono">
            {isEn ? currentTabInfo.subtitleEn : currentTabInfo.subtitleVi}
          </p>
        </section>

        {/* Main Tab Content */}
        <main className="flex-grow w-full px-4 sm:px-8 py-6 relative">
          <AnimatePresence mode="wait">
            {activeTab === 'theory' && (
              <motion.div
                key="theory"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                <TheoryTab />
              </motion.div>
            )}

            {activeTab === 'comparison' && (
              <motion.div
                key="comparison"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                <Suspense fallback={<TabFallback isEn={isEn} />}>
                  <SNNComparison />
                </Suspense>
              </motion.div>
            )}

            {activeTab === 'neuron' && (
              <motion.div
                key="neuron"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                <Suspense fallback={<TabFallback isEn={isEn} />}>
                  <NeuronTab />
                </Suspense>
              </motion.div>
            )}

            {activeTab === 'network' && (
              <motion.div
                key="network"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                <Suspense fallback={<TabFallback isEn={isEn} />}>
                  <NetworkTab />
                </Suspense>
              </motion.div>
            )}

            {activeTab === 'vision' && (
              <motion.div
                key="vision"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                <Suspense fallback={<TabFallback isEn={isEn} />}>
                  <VisionSNN onNavigateToHardware={() => setActiveTab('hardware')} />
                </Suspense>
              </motion.div>
            )}

            {activeTab === 'hardware' && (
              <motion.div
                key="hardware"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                <Suspense fallback={<TabFallback isEn={isEn} />}>
                  <NeuromorphicHardwareLab />
                </Suspense>
              </motion.div>
            )}

            {activeTab === 'pytorch' && (
              <motion.div
                key="pytorch"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.15 }}
              >
                <Suspense fallback={<TabFallback isEn={isEn} />}>
                  <PytorchTab />
                </Suspense>
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {/* Quiet Editorial Footer */}
        <footer className="border-t border-slate-800/60 bg-elegant-card/30 py-4 px-4 sm:px-8 mt-auto">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-elegant-muted font-mono">
            <span>Complex-Valued SNN Neuromorphic Workbench</span>
            <span>KAIST Memristor Co-Design · PyTorch torch.complex64</span>
          </div>
        </footer>
      </div>

      {/* PERSISTENT SUBTLE PERFORMANCE MONITOR OVERLAY */}
      <PerformanceMonitor activeTab={activeTab as ActiveModelTabId} />
    </div>
  );
}
