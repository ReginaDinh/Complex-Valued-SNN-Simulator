import os
import re
import json
import urllib.request
import urllib.parse
import time

API_KEY = os.environ.get("GEMINI_API_KEY")

def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    for _ in range(5):
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"Error: {e}, waiting 5s...")
            time.sleep(5)
    return "{}"

def has_vietnamese(text):
    vi_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
    return any(c in text for c in vi_chars)

with open('vi_strings.json', 'r') as f:
    vi_strings = json.load(f)

# Keep only those that don't have isEn or braces
vi_strings = [s for s in vi_strings if '{' not in s and '}' not in s]
vi_strings = sorted(list(set(vi_strings)), key=len, reverse=True)

translations = {}
chunk_size = 20
for i in range(0, len(vi_strings), chunk_size):
    chunk = vi_strings[i:i+chunk_size]
    prompt = f"Translate the following Vietnamese strings to English. Return ONLY a valid JSON object where the key is the EXACT original Vietnamese string and the value is the English translation. Keep technical terms intact.\n\n" + json.dumps(chunk, ensure_ascii=False)
    print(f"Calling Gemini Flash for chunk {i//chunk_size}...")
    response_json = call_gemini(prompt)
    
    try:
        res = json.loads(response_json)
        translations.update(res)
    except Exception as e:
        print(f"Failed to parse translations: {e}")
    time.sleep(2)

with open('translations.json', 'w') as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

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

for filename in files:
    with open(filename, 'r') as f:
        code = f.read()
        
    for vi, en in translations.items():
        if vi in code:
            en_safe = en.replace("'", "\\'").replace("\n", " ")
            code = code.replace(f">{vi}<", f">{{isEn ? '{en_safe}' : '{vi}'}}")
            code = code.replace(f'"{vi}"', f"{{isEn ? '{en_safe}' : '{vi}'}}")
            # Only do the simple text node replacements if not inside a tag attribute
            # We already handled >...< and "..."

    with open(filename, 'w') as f:
        f.write(code)
    print(f"Processed {filename}")

