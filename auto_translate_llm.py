import os
import re
import json
import urllib.request
import urllib.parse

API_KEY = os.environ.get("GEMINI_API_KEY")

def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={API_KEY}"
    
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
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Error: {e}")
        return "{}"

def has_vietnamese(text):
    vi_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
    return any(c in text for c in vi_chars)

files = [
    'src/components/SNNComparison.tsx', 
    'src/components/ChoiLabTab.tsx', 
    'src/components/VisionSNN.tsx',
    'src/components/NetworkTab.tsx',
    'src/components/HardwareIntegrationModal.tsx'
]

for filename in files:
    with open(filename, 'r') as f:
        code = f.read()
        
    matches = re.findall(r'>([^<]+)<', code)
    vi_strings = [m.strip() for m in matches if has_vietnamese(m.strip()) and not '{isEn' in m]
    
    if not vi_strings:
        continue
        
    # Deduplicate and sort by length descending to replace longest first
    vi_strings = sorted(list(set(vi_strings)), key=len, reverse=True)
    
    # Split into chunks if too many
    chunk_size = 50
    for i in range(0, len(vi_strings), chunk_size):
        chunk = vi_strings[i:i+chunk_size]
        prompt = f"Translate the following Vietnamese strings to English. Return a JSON object where the key is the EXACT original Vietnamese string and the value is the English translation. Keep technical terms intact.\n\n" + json.dumps(chunk, ensure_ascii=False)
        print(f"Calling Gemini for {filename} chunk {i//chunk_size}...")
        response_json = call_gemini(prompt)
        
        try:
            translations = json.loads(response_json)
            for vi, en in translations.items():
                # Avoid matching inside already formatted strings by using exact >...< or "..." match
                code = code.replace(f">{vi}<", f">{{isEn ? '{en}' : '{vi}'}}")
                code = code.replace(f'"{vi}"', f"{{isEn ? '{en}' : '{vi}'}}")
        except Exception as e:
            print(f"Failed to parse translations: {e}")

    with open(filename, 'w') as f:
        f.write(code)
    print(f"Processed {filename}")
