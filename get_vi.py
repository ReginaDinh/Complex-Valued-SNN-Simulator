import re
import json

def has_vietnamese(text):
    vi_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
    return any(c in text for c in vi_chars)

files = [
    'src/components/SNNComparison.tsx', 
    'src/components/ChoiLabTab.tsx', 
    'src/components/VisionSNN.tsx',
    'src/components/NetworkTab.tsx',
    'src/components/HardwareIntegrationModal.tsx',
    'src/components/TheoryTab.tsx',
    'src/components/NeuronTab.tsx',
    'src/components/PytorchTab.tsx'
]

translations = {}
for filename in files:
    with open(filename, 'r') as f:
        code = f.read()
    
    matches = re.findall(r'>([^<]+)<', code)
    for m in matches:
        s = m.strip()
        if has_vietnamese(s) and not '{isEn' in s:
            translations[s] = ""

print(json.dumps(translations, ensure_ascii=False, indent=2))
