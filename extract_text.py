import os
import re

files = [
    'src/components/TheoryTab.tsx',
    'src/components/SNNComparison.tsx',
    'src/components/PytorchTab.tsx',
    'src/components/NetworkTab.tsx',
    'src/components/VisionSNN.tsx',
    'src/components/ChoiLabTab.tsx',
    'src/components/HardwareIntegrationModal.tsx'
]

for file in files:
    with open(file, 'r') as f:
        content = f.read()
    # Simple heuristic to find Vietnamese text (contains words with accent marks)
    matches = re.findall(r'>([^<]*[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+[^<]*)<', content, re.IGNORECASE)
    if matches:
        print(f"--- {file} ---")
        for m in matches:
            if m.strip():
                print(m.strip())
