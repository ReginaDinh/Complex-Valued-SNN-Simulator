import re
import json
import glob

def extract():
    files = glob.glob('src/components/*.tsx')
    vi_strings = set()
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
        # Find all text between > and <
        matches = re.findall(r'>([^<]+)<', content)
        for m in matches:
            m = m.strip()
            if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', m, re.IGNORECASE):
                vi_strings.add(m)
        
        # Find strings in quotes (single, double, backticks)
        quote_matches = re.findall(r'["\'`]([^"\'`]+)["\'`]', content)
        for m in quote_matches:
            if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', m, re.IGNORECASE):
                vi_strings.add(m)

    print(json.dumps(list(vi_strings), indent=2, ensure_ascii=False))

extract()
