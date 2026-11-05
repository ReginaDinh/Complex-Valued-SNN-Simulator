with open('src/components/NeuronTab.tsx', 'r') as f:
    code = f.read()

if "useLanguage" not in code:
    code = code.replace("import { Activity, Play, Pause, RefreshCw, Sliders } from 'lucide-react';", "import { Activity, Play, Pause, RefreshCw, Sliders } from 'lucide-react';\nimport { useLanguage } from '../contexts/LanguageContext';")
    code = code.replace("export default function NeuronTab() {", "export default function NeuronTab() {\n  const { t } = useLanguage();")
    
    code = code.replace("Neuron Parameters", "{t('neuron.params')}")
    code = code.replace("Leak Magnitude:", "{t('neuron.leakMag')}")
    code = code.replace("Leak Phase (rad):", "{t('neuron.leakPhase')}")
    code = code.replace("Threshold:", "{t('neuron.threshold')}")
    code = code.replace("Input Current (Complex)", "{t('neuron.inputCurrent')}")
    code = code.replace("Amplitude:", "{t('neuron.amplitude')}")
    code = code.replace("Frequency:", "{t('neuron.frequency')}")
    code = code.replace("Complex Plane Trajectory", "{t('neuron.trajectory')}")
    
    with open('src/components/NeuronTab.tsx', 'w') as f:
        f.write(code)
