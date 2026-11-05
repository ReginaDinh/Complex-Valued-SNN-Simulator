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
    'src/components/NeuronTab.tsx'
]

vi_strings = []
for filename in files:
    with open(filename, 'r') as f:
        code = f.read()
    
    matches = re.findall(r'>([^<]+)<', code)
    strings = [m.strip() for m in matches if has_vietnamese(m.strip()) and not '{isEn' in m]
    vi_strings.extend(strings)

vi_strings = sorted(list(set(vi_strings)))
with open('vi_strings.json', 'w') as f:
    json.dump(vi_strings, f, ensure_ascii=False, indent=2)

print(f"Found {len(vi_strings)} strings.")
