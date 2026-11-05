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
        "So Sánh Kiến Trúc & Hoạt Động: SNN vs Complex-valued SNN (cLIF)": "Architecture & Operation Comparison: SNN vs Complex-valued SNN (cLIF)",
        "Khám phá sự khác biệt cốt lõi giữa Mạng nổ xung miền thực truyền thống (Traditional SNN) và Mạng nổ xung miền số phức (Complex-valued SNN).": "Explore the core differences between Traditional Real-valued Spiking Neural Networks (SNN) and Complex-valued Spiking Neural Networks (cLIF).",
        "Mạng Nổ Xung Miền Thực (Traditional SNN)": "Traditional Real-valued SNN",
        "Miền Thực (ℝ)": "Real Domain (ℝ)",
        "Mạng nổ xung truyền thống mô phỏng sinh học bằng các biến trạng thái là các số thực đơn thuần. Thông tin được mã hóa dựa trên ": "Traditional SNNs simulate biology using state variables that are simple real numbers. Information is encoded based on ",
        "tần số nổ xung (Rate coding)": "spike frequency (Rate coding)",
        " hoặc ": " or ",
        "thời điểm nổ xung đơn lẻ (Temporal coding)": "single spike timing (Temporal coding)",
        "■ Điện thế màng:": "■ Membrane Potential:",
        "V(t) ∈ ℝ — Một giá trị vô hướng (scalar) 1 chiều biểu diễn mức tích lũy năng lượng.": "V(t) ∈ ℝ — A 1-dimensional scalar value representing accumulated energy.",
        "■ Trọng số Synapse:": "■ Synaptic Weight:",
        "■ Tích lũy dòng điện:": "■ Current Accumulation:",
        "Phép cộng đại số trực tiếp các xung kích hoạt đi vào. Không có khái niệm lệch pha hay đồng bộ hóa.": "Direct algebraic addition of incoming activation spikes. No concept of phase shift or synchronization.",
        "■ Suy hao rò rỉ:": "■ Leakage Decay:",
        "Decay theo hàm mũ thẳng về trạng thái nghỉ 0 theo một chiều duy nhất.": "Exponential decay straight to resting state 0 in a single dimension.",
        "Mạng Nổ Xung Miền Phức (Complex-valued cLIF)": "Complex-valued SNN (cLIF)",
        "Miền Phức (ℂ)": "Complex Domain (ℂ)",
        "CV-SNN mở rộng trạng thái nổ xung sang miền phức để mô phỏng ": "CV-SNN extends spiking states to the complex domain to simulate ",
        "đồng bộ hóa pha và nhịp sinh học tự nhiên": "phase synchronization and natural biological rhythms",
        ". Thông tin mang cả biên độ (năng lượng) lẫn góc pha (thời gian tương đối).": ". Information carries both amplitude (energy) and phase angle (relative timing).",
        "U(t) = V(t) + j·Y(t) ∈ ℂ — Đại lượng 2 chiều mang cả biên độ |U(t)| và pha góc θ(t).": "U(t) = V(t) + j·Y(t) ∈ ℂ — A 2D quantity carrying both amplitude |U(t)| and phase angle θ(t).",
        "W = M·e^(jϕ) ∈ ℂ — Vừa điều khiển biên độ truyền M, vừa làm ": "W = M·e^(jϕ) ∈ ℂ — Controls both transmission amplitude M and causes ",
        "lệch pha dòng điện": "current phase shift",
        " một góc ϕ.": " by an angle ϕ.",
        "Phép cộng vector số phức, cho phép ": "Complex vector addition, allowing ",
        "Giao thoa cộng pha": "Constructive interference",
        " (constructive) hoặc ": " or ",
        "Giao thoa triệt tiêu": "Destructive interference",
        " (destructive) vi sai.": " (differential).",
        "Decay biên độ đồng thời xoay pha liên tục, vẽ nên một quỹ đạo xoắn ốc (spiral trajectory) tuyệt đẹp.": "Amplitude decay simultaneously rotates phase continuously, drawing a beautiful spiral trajectory.",
        "Mô Phỏng 1: Sự Kỳ Diệu Của Giao Thoa Pha (Phase Interference)": "Simulation 1: The Magic of Phase Interference",
        "Cơ chế cộng hợp vector vi sai": "Differential Vector Addition Mechanism",
        "Bảng So Sánh Chi Tiết (SNN vs cLIF)": "Detailed Comparison Table (SNN vs cLIF)",
        "Tính chất": "Property",
        "Không gian Trạng thái": "State Space",
        "Cách mã hóa thông tin": "Information Encoding",
        "Mã hóa tần số (Rate) hoặc mốc thời gian tuyệt đối.": "Rate coding or absolute timing.",
        "Mã hóa cả Biên độ (Amplitude) và Pha tương đối (Phase Angle).": "Encodes both Amplitude and relative Phase Angle.",
        "Phép nhân vô hướng thực: $W \cdot S(t)$": "Real scalar multiplication: $W \cdot S(t)$",
        "Hiện tượng tích tụ điện thế": "Potential Accumulation Phenomenon",
        "Tích lũy tịnh tiến (đơn thuần tăng/giảm biên độ).": "Translational accumulation (simply increasing/decreasing amplitude).",
        "Giao thoa sóng pha (Cộng hưởng hoặc Triệt tiêu vi sai).": "Phase wave interference (Resonance or differential Cancellation).",
        "Độ bền chống nhiễu dẫn nạp": "Conductance Noise Robustness",
        "Kém. Sai lệch conductance vật lý làm lệch hướng tính toán 1D trực tiếp.": "Poor. Physical conductance variations directly skew 1D computation.",
        "Cực tốt. Sai lệch dẫn nạp memristor PCM phân tán đều 360°, tự triệt tiêu ngẫu nhiên.": "Excellent. PCM memristor conductance variations are uniformly distributed over 360°, self-canceling randomly.",
        "Khả năng ánh xạ memristor": "Memristor Mapping Capability",
        "Cần nhiều linh kiện vi sai ghép cặp phức tạp để giả lập số âm.": "Requires many complex paired differential components to simulate negative numbers.",
        "Tự nhiên. Ánh xạ trực tiếp góc dẫn nạp vật lý thành Pha và Biên độ của mảng PCM.": "Natural. Directly maps physical conductance angle into Phase and Amplitude of the PCM array.",
        "Xung Vào": "Input Spike",
        "Xung 1D": "1D Spike",
        "Trọng Số": "Weight",
        "Trọng Số Thực": "Real Weight",
        "Ngưỡng Tròn": "Circular Threshold",
        "Ngưỡng 1D": "1D Threshold",
        "Đầu ra Memristor Core": "Memristor Core Output",
        "Tổng hợp Vi sai 1D": "1D Differential Sum",
        "Đầu vào kênh A. Trong SNN truyền thống là Xung 1D, trong CV-SNN là Vector.": "Channel A Input. In traditional SNN it's a 1D Spike, in CV-SNN it's a Vector.",
        "Đầu vào kênh B.": "Channel B Input.",
        "Tương đương với việc nhân trọng số (weight). SNN nhân số thực, CV-SNN nhân số phức (có xoay pha).": "Equivalent to weight multiplication. SNN uses real numbers, CV-SNN uses complex numbers (phase rotation).",
        "Màng Soma. Nơi tích tụ điện thế từ các kênh (Giao thoa).": "Soma Membrane. Where potential from channels accumulates (Interference).",
        "Sự phân rã (Leakage).": "Leakage decay.",
        "Cơ chế tạo xung đầu ra (Spike). SNN so sánh số thực, CV-SNN so sánh Vector.": "Output Spike generation mechanism. SNN compares real numbers, CV-SNN compares Vectors.",
    }

    for vi, en in translations.items():
        code = code.replace(f">{vi}<", f">{{isEn ? '{en}' : '{vi}'}}")
        code = code.replace(f'"{vi}"', f"{{isEn ? '{en}' : '{vi}'}}")

    with open(filename, 'w') as f:
        f.write(code)

for file in files:
    translate_file(file)

