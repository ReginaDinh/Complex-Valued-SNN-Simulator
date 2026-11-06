import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, Terminal, Cpu, Check, Copy, Info, AlertCircle } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import { useCopyStatus } from '../utils/useCopyStatus';

interface HardwareIntegrationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function HardwareIntegrationModal({ isOpen, onClose }: HardwareIntegrationModalProps) {
  const { status: copyStatus, copy } = useCopyStatus();
  const { language } = useLanguage();
  const isEn = language === "en";

  const pythonCode = `import numpy as np
import time
import pyvisa # ${isEn ? 'Communicate via VISA protocol with measurement devices (e.g. Agilent B1500A)' : 'Giao tiếp qua giao thức VISA với thiết bị đo (ví dụ: Agilent B1500A)'}

# ${isEn ? 'NOTE: This is pseudocode illustrating the process using Python' : 'LƯU Ý: Đây là mã giả (pseudocode) minh họa quy trình dùng Python'} 
# ${isEn ? 'to control Pulse Generator and SMU via GPIB/USB' : 'điều khiển máy phát xung (Pulse Generator) và SMU qua chuẩn GPIB/USB'}
# ${isEn ? 'to program physical weights into Memristor chips.' : 'để nạp trọng số vật lý vào chip Memristor.'}

print(${isEn ? '"Initializing connection with measurement equipment..."' : '"Đang khởi tạo kết nối với thiết bị đo lường..."'})
rm = pyvisa.ResourceManager()
instrument = rm.open_resource('USB0::0x0957::0x1736::MY51140135::0::INSTR')
instrument.write("*RST") # ${isEn ? 'Reset device' : 'Reset thiết bị'}

def apply_write_pulse(device_address, voltage_v, pulse_width_ns):
    # ${isEn ? 'Simulated SCPI command to generate voltage pulse' : 'Lệnh SCPI giả lập để phát xung điện áp'}
    instrument.write(f"SOUR:VOLT:LEV {voltage_v}")
    instrument.write(f"SOUR:PULS:WIDT {pulse_width_ns}e-9")
    instrument.write(f"ROUT:CLOS (@{device_address})")
    instrument.write("OUTP ON")
    time.sleep(0.01)
    instrument.write("OUTP OFF")

# ${isEn ? '2. Define complex weight matrix (from digital Simulator)' : '2. Định nghĩa ma trận trọng số phức (từ Simulator số)'}
# ${isEn ? 'Example: W_re_plus, W_re_minus, W_im_plus, W_im_minus for 1 layer' : 'Ví dụ: W_re_plus, W_re_minus, W_im_plus, W_im_minus cho 1 lớp'}
# ${isEn ? 'Format: NumPy array [64 inputs, 4 neurons]' : 'Định dạng: mảng NumPy [64 inputs, 4 neurons]'}
w_re = np.random.randn(64, 4) 
w_im = np.random.randn(64, 4)

# ${isEn ? '3. Compile and separate complex domain into 4 differential real matrices' : '3. Biên dịch phân tách miền phức thành 4 ma trận thực vi sai'}
G_re_plus  = np.maximum(w_re, 0)
G_re_minus = np.maximum(-w_re, 0)
G_im_plus  = np.maximum(w_im, 0)
G_im_minus = np.maximum(-w_im, 0)

# ${isEn ? '4. Map Conductance (G) to physical array' : '4. Ánh xạ Conductance (G) vào mảng vật lý'}
def program_crossbar_layer(layer_id, G_pos, G_neg):
    print(f${isEn ? '"Starting programming (Program/Set/Reset) on Crossbar {layer_id}..."' : '"Bắt đầu nạp (Program/Set/Reset) lên Crossbar {layer_id}..."'})
    for out_idx in range(4):
        for in_idx in range(64):
            # ${isEn ? 'Configure V_set / V_reset voltage levels via DAC' : 'Cấu hình mức điện áp V_set / V_reset thông qua DAC'}
            controller.program_cell(in_idx, out_idx, target_G=G_pos[in_idx, out_idx])
            controller.program_cell(in_idx + 64, out_idx, target_G=G_neg[in_idx, out_idx])
            time.sleep(0.001)

program_crossbar_layer('Real', G_re_plus, G_re_minus)
program_crossbar_layer('Imag', G_im_plus, G_im_minus)

# ${isEn ? '5. Run Inference' : '5. Chạy suy luận (Inference)'}
# ${isEn ? 'Represent phase signals as voltage pulses (V) with different Delays' : 'Biểu diễn tín hiệu pha thành các xung điện áp (V) với độ trễ (Delay) khác nhau'}
v_input_spikes = controller.generate_phase_encoded_spikes(input_image_array, freq=1.5)
current_outputs = controller.read_output_currents(v_input_spikes)

print(${isEn ? '"ON-CHIP INFERENCE RESULTS:"' : '"KẾT QUẢ SUY LUẬN TRÊN CHIP:"'})
print(current_outputs)

# =====================================================================
# ${isEn ? 'ON-CHIP LEARNING / HARDWARE-IN-THE-LOOP TRAINING' : 'TỰ HỌC TRÊN CHIP (ON-CHIP LEARNING / HARDWARE-IN-THE-LOOP TRAINING)'}
# =====================================================================
# ${isEn ? 'Instead of just programming pre-trained weights for inference, the algorithm proceeds to' : 'Thay vì chỉ nạp (inference) trọng số đã train sẵn, thuật toán tiến hành'}
# ${isEn ? 'update conductance by training directly on the physical device, overcoming' : 'huấn luyện cập nhật độ dẫn nạp ngay trên thiết bị thực tế, vượt qua'} 
# ${isEn ? 'the discrepancy between software simulation and physical reality (Device Variability).' : 'sự khác biệt giữa mô phỏng phần mềm và bản chất vật lý (Device Variability).'}

def on_chip_training_step(input_img, target_idx, learning_rate=0.01):
    # ${isEn ? '1. Forward Pass to measure actual output current' : '1. Lan truyền tiến (Forward Pass) để đo dòng điện đầu ra thực tế'}
    v_spikes = input_img # ${isEn ? 'Simulated signal' : 'Giả lập tín hiệu'}
    i_outs = np.random.randn(4) * 1e-6 # ${isEn ? 'Simulated measured current' : 'Giả lập dòng điện đo được'}
    
    # ${isEn ? '2. Calculate Error Vector on Host PC' : '2. Tính toán Vector Lỗi (Error) tại Host PC'}
    # ${isEn ? 'Target: 100uA for correct neuron, 0uA for wrong neurons' : 'Đích (Target): 100uA cho nơ-ron đúng, 0uA cho các nơ-ron sai'}
    targets = np.zeros(4)
    targets[target_idx] = 100e-6 
    errors = i_outs - targets
    
    # ${isEn ? '3. Calculate Gradient matrix (Delta G)' : '3. Tính ma trận Gradient thay đổi (Delta G)'}
    delta_W_re = np.zeros((64, 4))
    delta_W_im = np.zeros((64, 4))
    
    for in_idx, val in enumerate(input_img):
        # ${isEn ? 'Transform pixel values into complex domain based on phase coding' : 'Biến đổi giá trị pixel thành miền phức dựa theo hệ mã hóa pha'}
        input_phase = 0.5 # ${isEn ? 'Simulated phase' : 'Giả lập phase'} 
        input_val_re = val * np.cos(input_phase)
        input_val_im = val * np.sin(input_phase)
        
        for out_idx in range(4):
            # ${isEn ? 'Basic gradient based on Delta rule' : 'Gradient cơ bản dựa trên luật Delta'}
            delta_W_re[in_idx, out_idx] = -learning_rate * errors[out_idx] * input_val_re
            delta_W_im[in_idx, out_idx] = -learning_rate * errors[out_idx] * input_val_im
            
    # ${isEn ? '4. Send Update Spikes directly to Memristor Crossbar' : '4. Phát xung cập nhật (Update Spikes) trực tiếp vào Crossbar Memristor'}
    # apply_write_pulse(...) # ${isEn ? 'Call actual pulse generation function' : 'Gọi hàm phát xung thực tế'}
    
    return np.mean(errors**2) # Loss

# ${isEn ? 'Execute actual loop training' : 'Thực thi huấn luyện vòng lặp thực tế'}
print(${isEn ? '"\\n--- Starting physical training (Hardware-in-the-Loop) ---"' : '"\\n--- Bắt đầu huấn luyện vật lý (Hardware-in-the-Loop) ---"'})
for epoch in range(10):
    total_loss = 0
    # ${isEn ? 'Assume training_dataset contains images and labels' : 'Giả sử training_dataset đã chứa ảnh và nhãn'}
    for img, label in training_dataset:
        total_loss += on_chip_training_step(img, label)
    print(f${isEn ? '"Epoch {epoch+1}/10 | Physical Network Loss: {total_loss:.4e}"' : '"Epoch {epoch+1}/10 | Tổn thất Mạng vật lý: {total_loss:.4e}"'})
`;

  const copyToClipboard = () => {
    void copy(pythonCode);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-4xl bg-elegant-bg border border-elegant-muted/20 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-elegant-muted/10 bg-elegant-card/50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-elegant-cyan/10 rounded-lg text-elegant-cyan">
                  <Cpu className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white font-display">{isEn ? 'Hardware Integration (API)' : 'Tích Hợp Phần Cứng (Hardware API)'}</h3>
                  <p className="text-xs text-elegant-muted font-mono mt-0.5">{isEn ? 'Python code for connecting & programming weights to test board' : 'Mã Python kết nối & Nạp trọng số lên bo mạch kiểm thử'}</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 text-elegant-muted hover:text-white hover:bg-elegant-card rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              
              <div className="bg-elegant-cyan/10 border border-elegant-cyan/20 p-5 rounded-xl space-y-4">
                <h4 className="text-sm font-bold text-elegant-cyan uppercase tracking-wider font-mono flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  {isEn ? 'For Beginners: Understanding Analog Chip Programming' : 'Dành Cho Người Mới: Hiểu Về Lập Trình Chip Analog'}
                </h4>
                <div className="space-y-3 text-[13px] text-elegant-text/90 leading-relaxed">
                  <p>
                    {isEn ? 'Programming neuromorphic/memristor chips is ' : 'Lập trình trên chip thần kinh nhân tạo (Neuromorphic/Memristor) '}<strong>{isEn ? 'very different' : 'rất khác'}</strong>{isEn ? ' from writing standard software code (C++, Python on CPU/GPU). Below are 3 core differences you need to know:' : ' so với viết code phần mềm thông thường (C++, Python chạy trên CPU/GPU). Dưới đây là 3 khác biệt cốt lõi bạn cần nắm:'}
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-1">
                    <li>
                      <strong className="text-white">{isEn ? 'Code does not run "inside" the chip:' : 'Code không chạy "bên trong" chip:'}</strong>{isEn ? ' Memristor chips are just physical resistor grids. The Python code below actually runs on your computer (Host PC) to control measurement equipment (like Oscilloscope/SMU) to pump voltage pulses into the chip pins.' : ' Chip Memristor chỉ là một mảng lưới các điện trở vật lý. Đoạn code Python dưới đây thực chất chạy trên máy tính của bạn (Host PC), dùng để điều khiển thiết bị đo lường (như máy Oscilloscope/SMU) bơm các xung điện áp (Voltage) vào chân của con chip.'}
                    </li>
                    <li>
                      <strong className="text-white">{isEn ? 'No "Flashing" concept:' : 'Không có khái niệm "Đổ Code" (Flashing):'}</strong>{isEn ? ' You do not compile code (like C++) and flash it to the Memristor\'s Flash memory as with Arduino/Microcontrollers. The "API" here is ' : ' Bạn không compile code (như C++) rồi nạp vào bộ nhớ Flash của Memristor như làm với Arduino/Vi điều khiển. "API" ở đây là '}<i>Hardware Control API</i>{isEn ? ' (like a library ' : ' (như thư viện '}<code>pyvisa</code>{isEn ? ') running on a PC, commanding Pulse Generators to send physical voltage streams into each pixel on the chip to alter its atomic structure (program weights).' : ') chạy trên máy tính, ra lệnh cho các máy phát xung (Pulse Generator) bắn các luồng điện áp vật lý vào từng điểm ảnh trên chip để thay đổi cấu trúc nguyên tử của nó (nạp trọng số).'}
                    </li>
                    <li>
                      <strong className="text-white">{isEn ? 'Variables are Physical States:' : 'Biến số là Trạng thái Vật lý:'}</strong>{isEn ? ' In software code, you assign ' : ' Trong code phần mềm, bạn gán '}<code>w = 0.5</code>{isEn ? '. On this chip, you use an API to deliver a Set/Reset voltage pulse strong enough to alter the crystal structure of the material (changing electrical conductance). "Saving data" is modifying component resistance.' : '. Trên chip này, bạn dùng API để phát một xung điện áp Set/Reset đủ mạnh làm biến đổi cấu trúc tinh thể của vật liệu (thay đổi độ dẫn điện - Conductance). Việc "lưu dữ liệu" chính là thay đổi điện trở của linh kiện.'}
                    </li>
                    <li>
                      <strong className="text-white">{isEn ? 'Calculation by Ohm\'s Law:' : 'Phép tính bằng Định luật Ôm:'}</strong>{isEn ? ' Instead of the CPU computing multiplication ' : ' Thay vì CPU phải tính toán phép nhân '}<code>A * B</code>{isEn ? ', we just need to supply voltage ' : ', ta chỉ cần cấp điện áp '}<strong>V</strong>{isEn ? ' to the input. The Memristor device has conductance ' : ' vào đầu vào, thiết bị Memristor có độ dẫn '}<strong>G</strong>{isEn ? ', so the output current automatically becomes ' : ', dòng điện sinh ra ở đầu ra sẽ tự động là '}<strong>I = V * G</strong>{isEn ? ' (by Ohm\'s law). Matrix addition is computed instantly via Kirchhoff\'s law (currents summing at a node).' : ' (theo định luật Ohm). Phép cộng ma trận được tính tức thời bằng định luật Kirchhoff (các dòng điện gộp lại tại một nút).'}
                    </li>
                  </ul>
                </div>
              </div>

              <div className="bg-[#2A1B1B] border border-rose-500/20 p-5 rounded-xl space-y-4">
                <h4 className="text-sm font-bold text-rose-400 uppercase tracking-wider font-mono flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  {isEn ? 'Harsh Reality: Why isn\'t this technology widespread yet?' : 'Thực tế phũ phàng: Tại sao công nghệ này chưa phổ biến?'}
                </h4>
                <div className="space-y-3 text-[13px] text-elegant-text/90 leading-relaxed">
                  <p>
                    {isEn ? 'Very perceptive observation! If everything were as perfect as this simulation, analog AI chips would have already dominated the world. In reality, Prof. Choi\'s lab (specializing in hardware components) and materials scientists struggle with extremely harsh physical limits:' : 'Bạn nhận xét rất tinh tế! Nếu mọi thứ hoàn hảo như mô phỏng này, AI chip analog đã thống trị thế giới. Thực tế, phòng lab của GS. Choi (chuyên về phần cứng linh kiện) và các nhà khoa học vật liệu phải vật lộn với các giới hạn vật lý cực kỳ khắc nghiệt:'}
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-1">
                    <li>
                      <strong className="text-white">{isEn ? 'Variability & Noise:' : 'Độ biến thiên (Variability) & Nhiễu:'}</strong> {isEn ? 'No two memristors are 100% identical. With the same 1.5V programming pulse, point A might reach 100Ω while point B reaches 120Ω. Furthermore, over time these values drift.' : 'Không có 2 memristor nào giống nhau 100%. Cùng một xung điện áp nạp 1.5V, điện trở của điểm A có thể đạt 100Ω, nhưng điểm B lại ra 120Ω. Chưa kể theo thời gian, các giá trị này bị trôi (drift).'}
                    </li>
                    <li>
                      <strong className="text-white">{isEn ? 'Sneak Path Problem:' : 'Dòng rò (Sneak Path Problem):'}</strong> {isEn ? 'In a crossbar matrix, current does not only pass through one intersection but leaks backward through neighboring components, distorting the entire calculation.' : 'Trong cấu trúc lưới Crossbar, dòng điện không chỉ đi qua 1 điểm giao cắt mà "rò rỉ" ngược qua các linh kiện lân cận, làm sai lệch toàn bộ kết quả tính toán.'}
                    </li>
                    <li>
                      <strong className="text-white">{isEn ? 'ADC/DAC Bottleneck:' : 'Nút thắt ADC/DAC:'}</strong> {isEn ? 'Although computing on the analog core saves 1000x energy, converting digital signals (from image on PC) to analog (DAC) for chip input, then converting analog back to digital (ADC) to read results consumes up to 90% of total system area and power!' : 'Mặc dù tính toán trên lõi analog tiết kiệm năng lượng 1000 lần, nhưng việc chuyển đổi tín hiệu số (từ ảnh trên máy tính) sang analog (DAC) để nạp vào chip, rồi lại chuyển analog thành số (ADC) để đọc kết quả... lại tiêu tốn đến 90% diện tích và năng lượng của toàn hệ thống!'}
                    </li>
                  </ul>
                  <p className="mt-2 text-rose-300 italic">
                    {isEn ? '* Scientific Note: Prof. Choi\'s (KAIST) research focuses on the ' : '* Đính chính khoa học: Nghiên cứu của GS. Choi (KAIST) tập trung vào '}<strong>{isEn ? 'hardware core (Memristor array)' : 'cốt lõi phần cứng (Memristor array)'}</strong>{isEn ? ' and physical self-correction/write-verify mechanisms. While ' : ' và cơ chế tự bù lỗi vật lý (self-correction/write-verify). Còn '}<strong>CV-SNN (Complex-Valued SNN)</strong>{isEn ? ' is a separate machine learning algorithm branch. In this simulation, we combine both: taking an advanced algorithm (CV-SNN) mapped onto KAIST\'s analog chip architecture to show the intersection of phase algorithms and quantum/analog hardware.' : ' là một nhánh thuật toán học máy riêng biệt. Trong mô phỏng này, chúng ta đang kết hợp cả hai: Lấy một thuật toán tiên tiến (CV-SNN) ánh xạ xuống nền tảng kiến trúc chip analog của KAIST để thấy sự giao thoa giữa thuật toán pha và phần cứng lượng tử/analog.'}
                  </p>
                </div>
              </div>

              <div className="bg-elegant-card border border-elegant-muted/10 p-5 rounded-xl space-y-3">
                <h4 className="text-sm font-bold text-white uppercase tracking-wider font-mono">{isEn ? 'Workflow' : 'Quy Trình Hoạt Động (Workflow)'}</h4>
                <ul className="text-sm text-elegant-text/80 space-y-2 list-disc list-inside">
                  <li><strong>{isEn ? 'Phase 1:' : 'Giai đoạn 1:'}</strong>{isEn ? ' Train using the software simulator on the main screen (Software Compiler) to optimize weights.' : ' Huấn luyện bằng công cụ giả lập phần mềm ở màn hình chính (Software Compiler) để tối ưu hóa trọng số.'}</li>
                  <li><strong>{isEn ? 'Phase 2:' : 'Giai đoạn 2:'}</strong>{isEn ? ' Export the weight matrix (G_re+, G_re-, G_im+, G_im-) from the software.' : ' Xuất ma trận trọng số (G_re+, G_re-, G_im+, G_im-) từ phần mềm.'}</li>
                  <li><strong>{isEn ? 'Phase 3:' : 'Giai đoạn 3:'}</strong>{isEn ? ' Use GPIB/VISA or Serial protocols through a semiconductor parameter analyzer (like Agilent/Keysight) to send programming voltage pulses (Set/Reset) to each cross-point on the KAIST chip.' : ' Sử dụng giao thức GPIB/VISA hoặc Serial thông qua bộ đo thông số bán dẫn (như Agilent/Keysight) để phát xung điện áp nạp (Set/Reset) vào từng điểm cầu (cross-point) trên chip KAIST.'}</li>
                  <li><strong>{isEn ? 'Phase 4:' : 'Giai đoạn 4:'}</strong>{isEn ? ' Enable the Self-correction mode on the board to fine-tune the conductance values accurately.' : ' Bật chế độ tự động bù lỗi (Self-correction) trên bo mạch để tinh chỉnh giá trị dẫn nạp chuẩn xác.'}</li>
                </ul>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-elegant-cyan text-sm font-mono font-bold">
                    <Terminal className="w-4 h-4" />
                    <span>{isEn ? 'Python Integration Code' : 'Mã Tích Hợp Python'}</span>
                  </div>
                  <button 
                    onClick={copyToClipboard}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-elegant-card hover:bg-elegant-card/80 border border-elegant-muted/20 rounded-md text-xs font-medium text-white transition-colors"
                  >
                    {copyStatus === 'copied' ? (
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                    ) : copyStatus === 'failed' ? (
                      <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    {copyStatus === 'copied'
                      ? isEn ? 'Copied' : 'Đã sao chép'
                      : copyStatus === 'failed'
                        ? isEn ? 'Copy failed: select manually' : 'Chép thất bại: hãy bôi đen để chép'
                        : isEn ? 'Copy code' : 'Sao chép mã'}
                  </button>
                </div>
                
                <div className="relative rounded-xl overflow-hidden bg-[#1a1b26] border border-elegant-muted/20">
                  <pre className="p-5 text-[13px] font-mono leading-relaxed overflow-x-auto text-gray-300">
                    <code dangerouslySetInnerHTML={{ __html: highlightPython(pythonCode) }} />
                  </pre>
                </div>
              </div>

            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

// Single-pass Python highlighter: tokens are matched once on the RAW source,
// every piece of text is HTML-escaped, so inserted <span> tags are never re-matched.
const PY_KEYWORDS = new Set([
  'import', 'from', 'as', 'def', 'class', 'return', 'for', 'while', 'in', 'if', 'elif', 'else',
  'try', 'except', 'finally', 'with', 'lambda', 'pass', 'break', 'continue', 'raise', 'yield',
  'and', 'or', 'not', 'is', 'None', 'True', 'False', 'print',
]);
const PY_KNOWN_NAMES = new Set([
  'np', 'pyvisa', 'time', 'controller', 'rm', 'instrument',
  'G_re_plus', 'G_re_minus', 'G_im_plus', 'G_im_minus', 'target_G',
]);

const PY_TOKEN_RE =
  /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|\b([A-Za-z_]\w*)\b|\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/g;

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightPython(code: string): string {
  let out = '';
  let last = 0;
  for (const m of code.matchAll(PY_TOKEN_RE)) {
    const start = m.index ?? 0;
    const [token, comment, str, ident, num] = m;
    out += escapeHtml(code.slice(last, start));
    const esc = escapeHtml(token);
    if (comment) out += `<span class="text-gray-500 italic">${esc}</span>`;
    else if (str) out += `<span class="text-emerald-400">${esc}</span>`;
    else if (ident && PY_KEYWORDS.has(ident)) out += `<span class="text-pink-400 font-bold">${esc}</span>`;
    else if (ident && PY_KNOWN_NAMES.has(ident)) out += `<span class="text-blue-400">${esc}</span>`;
    else if (num) out += `<span class="text-orange-400">${esc}</span>`;
    else out += esc;
    last = start + token.length;
  }
  return out + escapeHtml(code.slice(last));
}