import re

files = [
    'src/components/SNNComparison.tsx', 
    'src/components/ChoiLabTab.tsx', 
    'src/components/VisionSNN.tsx',
    'src/components/NetworkTab.tsx',
    'src/components/TheoryTab.tsx'
]

def translate_file(filename):
    with open(filename, 'r') as f:
        code = f.read()

    translations = {
        "Thử Nghiệm Thị Giác Máy Tính & Biên Dịch Phần Cứng CV-SNN": "Computer Vision Experiment & CV-SNN Hardware Compiler",
        "Phần Mềm Đã Đồng Bộ Hóa": "Software Synchronized",
        "1. Bản Vẽ & Mã Hóa Pha (Drawing Canvas)": "1. Drawing Canvas & Phase Encoding",
        "Xóa Canvas": "Clear Canvas",
        "góc pha (θ)": "phase angle (θ)",
        "Mẫu chuẩn lưu sẵn (Templates Presets):": "Template Presets:",
        "Ký tự {name}": "Character {name}",
        "Phương pháp Mã hóa Pha (Phase Mapping):": "Phase Mapping Method:",
        "Hệ số nhân góc pha (γ):": "Phase Multiplier (γ):",
        "2. Bộ Biên Dịch & Ánh Xạ Chip Memristor": "2. Memristor Chip Mapping Compiler",
        "Bộ Biên Dịch Ánh Xạ (Hardware Mapping Compiler)": "Hardware Mapping Compiler",
        "Độ phân giải Memristor PCM:": "PCM Memristor Resolution:",
        "Độ nhiễu nạp ghi (Write/Conductance Noise):": "Write/Conductance Noise:",
        "Cơ chế On-Chip Self-Correction (KAIST 2024)": "On-Chip Self-Correction Mechanism (KAIST 2024)",
        "Chỉ số Hiệu năng Ước tính trên Chip:": "Estimated On-Chip Performance Metrics:",
        "Mã Lệnh Tích Hợp Phần Cứng (API)": "Hardware Integration API Code",
        "3. Kết quả Nhận diện Miền Phức (Live Inference)": "3. Complex Domain Live Inference Results",
        "KẾT QUẢ PHÂN LOẠI HIỆN TẠI:": "CURRENT CLASSIFICATION RESULT:",
        "Tại sao miền pha số phức của CV-SNN là chiếc \"Chìa Khóa Vàng\" để mở rộng ứng dụng thị giác máy tính?": "Why is the complex phase domain of CV-SNN the \"Golden Key\" to expanding computer vision applications?",
        "vị trí pixel thành các giá trị góc pha (θ) tương đối": "pixel positions into relative phase angle values (θ)",
        "phép tính cộng tích chập pha tự nhiên": "natural phase convolution addition calculations",
        "Trọng số Synapse": "Synaptic Weight",
        "Điện thế màng": "Membrane Potential",
        "Tích lũy dòng điện": "Current Accumulation",
        "Suy hao rò rỉ": "Leakage Decay",
        "Lập trình trên chip thần kinh nhân tạo (Neuromorphic/Memristor)": "Programming on artificial neuromorphic/memristor chips",
        "Giai đoạn 1:": "Phase 1:",
        "Giai đoạn 2:": "Phase 2:",
        "Giai đoạn 3:": "Phase 3:",
        "Giai đoạn 4:": "Phase 4:",
        "Quy Trình Hoạt Động (Workflow)": "Operational Workflow",
        "Mô Phỏng SNN": "SNN Simulation",
        "Tạm dừng": "Pause",
        "Chạy tiếp": "Resume",
        "Ký tự \"{CLASS_NAMES[predictionIdx]}\"": "Character \"{CLASS_NAMES[predictionIdx]}\"",
    }

    for vi, en in translations.items():
        code = code.replace(f">{vi}<", f">{{isEn ? '{en}' : '{vi}'}}")
        code = code.replace(f'"{vi}"', f"{{isEn ? '{en}' : '{vi}'}}")

    with open(filename, 'w') as f:
        f.write(code)

for file in files:
    translate_file(file)

