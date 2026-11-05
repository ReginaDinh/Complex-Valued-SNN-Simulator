import re

with open('src/components/HardwareIntegrationModal.tsx', 'r') as f:
    code = f.read()

replaces = {
    "Tích Hợp Phần Cứng (Hardware API)": "Hardware Integration (API)",
    "Mã Python kết nối & Nạp trọng số lên bo mạch kiểm thử": "Python code for connecting & programming weights to test board",
    "Dành Cho Người Mới: Hiểu Về Lập Trình Chip Analog": "For Beginners: Understanding Analog Chip Programming",
    "rất khác": "very differently",
    "Code không chạy \"bên trong\" chip:": "Code does not run \"inside\" the chip:",
    "Không có khái niệm \"Đổ Code\" (Flashing):": "No \"Flashing\" concept:",
    " Bạn không compile code (như C++) rồi nạp vào bộ nhớ Flash của Memristor như làm với Arduino/Vi điều khiển. \"API\" ở đây là ": " You do not compile code (like C++) and flash it to the Memristor's Flash memory as with Arduino/Microcontrollers. The \"API\" here is ",
    " (như thư viện ": " (like a library ",
    "Biến số là Trạng thái Vật lý:": "Variables are Physical States:",
    " Trong code phần mềm, bạn gán ": " In software code, you assign ",
    "Phép tính bằng Định luật Ôm:": "Calculation by Ohm's Law:",
    " Thay vì CPU phải tính toán phép nhân ": " Instead of the CPU computing the multiplication ",
    ", ta chỉ cần cấp điện áp ": ", we just need to supply voltage ",
    " vào đầu vào, thiết bị Memristor có độ dẫn ": " to the input, the Memristor device has conductance ",
    ", dòng điện sinh ra ở đầu ra sẽ tự động là ": ", the current generated at the output will automatically be ",
    "Độ biến thiên (Variability) & Nhiễu:": "Variability & Noise:",
    "Dòng rò (Sneak Path Problem):": "Sneak Path Problem:",
    "Nút thắt ADC/DAC:": "ADC/DAC Bottleneck:",
    "cốt lõi phần cứng (Memristor array)": "hardware core (Memristor array)",
    " và cơ chế tự bù lỗi vật lý (self-correction/write-verify). Còn ": " and physical self-correction/write-verify mechanisms. While ",
    "Quy Trình Hoạt Động (Workflow)": "Workflow",
    "Giai đoạn 1:": "Phase 1:",
    " Huấn luyện bằng công cụ giả lập phần mềm ở màn hình chính (Software Compiler) để tối ưu hóa trọng số.": " Train using the software simulator on the main screen (Software Compiler) to optimize weights.",
    "Giai đoạn 2:": "Phase 2:",
    " Xuất ma trận trọng số (G_re+, G_re-, G_im+, G_im-) từ phần mềm.": " Export the weight matrix (G_re+, G_re-, G_im+, G_im-) from the software.",
    "Giai đoạn 3:": "Phase 3:",
    " Sử dụng giao thức GPIB/VISA hoặc Serial thông qua bộ đo thông số bán dẫn (như Agilent/Keysight) để phát xung điện áp nạp (Set/Reset) vào từng điểm cầu (cross-point) trên chip KAIST.": " Use GPIB/VISA or Serial protocols through a semiconductor parameter analyzer (like Agilent/Keysight) to send programming voltage pulses (Set/Reset) to each cross-point on the KAIST chip.",
    "Giai đoạn 4:": "Phase 4:",
    " Bật chế độ tự động bù lỗi (Self-correction) trên bo mạch để tinh chỉnh giá trị dẫn nạp chuẩn xác.": " Enable the Self-correction mode on the board to fine-tune the conductance values accurately.",
    "Sao chép mã": "Copy code",
    "Đã sao chép": "Copied"
}

if "useLanguage" not in code:
    code = code.replace("from 'lucide-react';", "from 'lucide-react';\nimport { useLanguage } from '../contexts/LanguageContext';")
    code = code.replace("export default function HardwareIntegrationModal({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) {", "export default function HardwareIntegrationModal({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) {\n  const { language } = useLanguage();\n  const isEn = language === 'en';")

for vi, en in replaces.items():
    code = code.replace(f">{vi}<", f">{{isEn ? '{en}' : '{vi}'}}")
    code = code.replace(f'"{vi}"', f"{{isEn ? '{en}' : '{vi}'}}")

with open('src/components/HardwareIntegrationModal.tsx', 'w') as f:
    f.write(code)

