export const pytorchCodeString = `import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class VonMisesSurrogate(torch.autograd.Function):
    """
    SOTA: Bessel-Normalized Von Mises Circular Surrogate Gradient for cLIF Neurons.
    Solves the non-differentiable circular threshold condition |U| >= V_th
    using directional statistics and Wirtinger complex calculus.
    """
    @staticmethod
    def forward(ctx, u: torch.Tensor, threshold: float, kappa: float = 2.5):
        magnitude = torch.abs(u)
        spikes_fired = (magnitude >= threshold).to(u.real.dtype)

        # Unit phasor e^{j * arg(u)} = u / (|u| + eps) avoids atan2 branch cut instability
        unit_phasor = u / (magnitude + 1e-8)
        ctx.save_for_backward(u, magnitude, unit_phasor, spikes_fired)
        ctx.threshold = threshold
        ctx.kappa = kappa

        # Emitted complex spike preserves instantaneous membrane phase: S(t) = H(|U| - V_th) * e^{j * theta}
        return spikes_fired.to(u.dtype) * unit_phasor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        u, magnitude, unit_phasor, spikes_fired = ctx.saved_tensors
        threshold = ctx.threshold
        kappa = ctx.kappa

        # Normalized radial distance from circular firing threshold
        diff = torch.clamp((magnitude - threshold) / max(threshold, 1e-6), -1.0, 1.0)

        # Normalized Von Mises radial surrogate density:
        # g'(diff) = exp(kappa * cos(pi * diff)) / (2 * pi * I_0(kappa))
        kappa_t = torch.tensor(kappa, dtype=magnitude.dtype, device=magnitude.device)
        i0_val = torch.special.i0(kappa_t)
        surrogate_radial = torch.exp(kappa * torch.cos(math.pi * diff)) / (2.0 * math.pi * i0_val + 1e-8)

        # Wirtinger derivative projection:
        # Radial component scales magnitude transition; tangential component preserves phase gradient
        radial_proj = torch.real(grad_output * torch.conj(unit_phasor))
        # Phase (tangential) gradient exists only where a spike was actually emitted:
        # for silent neurons the forward output is identically 0, so d(output)/d(phase) = 0.
        tangential_proj = spikes_fired * torch.imag(grad_output * torch.conj(unit_phasor)) / (magnitude + threshold)

        grad_u = (surrogate_radial * radial_proj + 1j * tangential_proj * 0.25) * unit_phasor
        return grad_u, None, None


class KaratsubaComplexLinear(nn.Module):
    """
    SOTA: True 3-Multiply Karatsuba Complex Linear Layer.
    Standard complex multiplication (W_r + j W_i)(X_r + j X_i) requires 4 real MatMuls.
    Karatsuba identity reduces this to 3 real MatMuls:
        k1 = W_r * X_r
        k2 = W_i * X_i
        k3 = (W_r + W_i) * (X_r + X_i)
        Re(Y) = k1 - k2 + b_r
        Im(Y) = k3 - k1 - k2 + b_i
    Achieves 25% reduction in multiplier array energy on neuromorphic hardware.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Complex Xavier / Glorot initialization
        scale = math.sqrt(1.5 / in_features)
        self.weight_re = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.weight_im = nn.Parameter(torch.randn(out_features, in_features) * scale)

        if bias:
            self.bias_re = nn.Parameter(torch.zeros(out_features))
            self.bias_im = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_re', None)
            self.register_parameter('bias_im', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.real, x.imag
        wr, wi = self.weight_re, self.weight_im

        # Exactly 3 real matrix multiplications (k1, k2, k3)
        k1 = F.linear(xr, wr)
        k2 = F.linear(xi, wi)
        k3 = F.linear(xr + xi, wr + wi)

        out_real = k1 - k2
        out_imag = k3 - k1 - k2

        if self.bias_re is not None:
            out_real = out_real + self.bias_re
            out_imag = out_imag + self.bias_im

        return torch.complex(out_real, out_imag)


class ComplexLIF(nn.Module):
    """
    Complex-Valued Leaky Integrate-and-Fire (cLIF) Neuron with Learnable Spiral Decay.
    Equations:
        1. U_pre[t] = beta * U[t-1] + I[t],  where beta = |beta| * e^{j * omega}
        2. S[t] = H(|U_pre[t]| - V_th) * e^{j * arg(U_pre[t])}
        3. Soft Reset: U[t] = U_pre[t] - V_th * S[t]
           Hard Reset: U[t] = U_pre[t] * (1 - |S[t]|)
    """
    def __init__(
        self,
        num_features: int,
        threshold: float = 1.0,
        leak_mag: float = 0.85,
        leak_phase: float = 0.15,
        reset_mechanism: str = 'soft',
    ):
        super().__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism

        # Learnable decay magnitude and intrinsic subthreshold oscillation frequency
        self.leak_mag = nn.Parameter(torch.full((num_features,), leak_mag, dtype=torch.float32))
        self.leak_phase = nn.Parameter(torch.full((num_features,), leak_phase, dtype=torch.float32))
        self.u = None

    def init_state(self, batch_size: int, device: torch.device):
        self.u = torch.zeros(batch_size, self.num_features, dtype=torch.complex64, device=device)

    def forward(self, input_current: torch.Tensor):
        if self.u is None or self.u.shape[0] != input_current.shape[0]:
            self.init_state(input_current.shape[0], input_current.device)

        mag = torch.clamp(self.leak_mag, 0.05, 0.99)
        beta = torch.polar(mag, self.leak_phase)

        # Pre-reset membrane potential integration
        u_pre = beta * self.u + input_current
        spikes = VonMisesSurrogate.apply(u_pre, self.threshold)

        # Post-spike membrane reset (detach spike mask on reset path for stable BPTT)
        if self.reset_mechanism == 'soft':
            self.u = u_pre - spikes.detach() * self.threshold
        else:
            fired_mask = (torch.abs(spikes.detach()) > 0)
            self.u = torch.where(fired_mask, torch.zeros_like(u_pre), u_pre)

        return spikes, u_pre


class PhaseSynchronyAttention(nn.Module):
    """
    SOTA: Temporal Phase-Synchrony Attention across time steps (T, B, C).
    Computes Hermitian inner product Re(Q * K^H) on unit phasors to avoid
    +-pi angle wrap-around discontinuities.
    """
    def __init__(self, features: int):
        super().__init__()
        self.q_proj = KaratsubaComplexLinear(features, features, bias=False)
        self.k_proj = KaratsubaComplexLinear(features, features, bias=False)
        self.v_proj = KaratsubaComplexLinear(features, features, bias=True)
        self.scale = 1.0 / math.sqrt(features)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq shape: (T, B, C) -> transpose to (B, T, C)
        z = z_seq.permute(1, 0, 2)
        q = self.q_proj(z)
        k = self.k_proj(z)
        v = self.v_proj(z)

        # Normalize Q and K to unit phasors so attention scores measure pure phase coherence
        q_unit = q / (torch.abs(q) + 1e-6)
        k_unit = k / (torch.abs(k) + 1e-6)

        # Hermitian phase-locking score: Re(Q @ K^H) / sqrt(C) -> shape (B, T, T)
        phase_scores = torch.real(torch.matmul(q_unit, torch.conj(k_unit).transpose(-2, -1))) * self.scale
        attn_weights = torch.softmax(phase_scores, dim=-1).to(v.dtype)

        out = torch.matmul(attn_weights, v)
        return (out + z).permute(1, 0, 2)


class ComplexSNN(nn.Module):
    """
    End-to-End Complex-Valued Spiking Neural Network with Karatsuba Synapses,
    Temporal Phase-Synchrony Attention, and Pre-Reset Membrane Readout.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = KaratsubaComplexLinear(in_dim, hidden_dim)
        self.lif1 = ComplexLIF(hidden_dim, threshold=1.0, leak_mag=0.85, leak_phase=0.15)
        self.phase_attn = PhaseSynchronyAttention(hidden_dim)
        self.fc2 = KaratsubaComplexLinear(hidden_dim, out_dim)
        self.lif2 = ComplexLIF(out_dim, threshold=1.0, leak_mag=0.85, leak_phase=0.12)

    def forward(self, x_seq: torch.Tensor):
        # x_seq: (time_steps, batch_size, in_dim) complex64
        time_steps, batch_size, _ = x_seq.shape
        self.lif1.init_state(batch_size, x_seq.device)
        self.lif2.init_state(batch_size, x_seq.device)

        hidden_spikes = []
        for t in range(time_steps):
            spk1, _ = self.lif1(self.fc1(x_seq[t]))
            hidden_spikes.append(spk1)

        hidden_seq = torch.stack(hidden_spikes, dim=0)  # (T, B, hidden_dim)
        attended_seq = self.phase_attn(hidden_seq)      # Phase-locked feature stream

        out_spikes = []
        out_potentials = []
        for t in range(time_steps):
            spk2, u_pre2 = self.lif2(self.fc2(attended_seq[t]))
            out_spikes.append(spk2)
            out_potentials.append(u_pre2)

        out_spikes_seq = torch.stack(out_spikes, dim=0)
        out_pot_seq = torch.stack(out_potentials, dim=0)

        # Hybrid readout: pre-reset membrane magnitude + firing rate
        readout_logits = torch.mean(torch.abs(out_pot_seq) + 0.5 * torch.abs(out_spikes_seq), dim=0)
        return readout_logits, out_spikes_seq


if __name__ == "__main__":
    batch_size, time_steps, in_features, hidden_features, out_features = 16, 30, 16, 32, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalized time axis t_norm in [0, 1) prevents Nyquist integer aliasing
    t_norm = (torch.arange(time_steps, device=device, dtype=torch.float32) / time_steps).view(time_steps, 1, 1)
    model = ComplexSNN(in_features, hidden_features, out_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(1, 6):
        targets = torch.randint(0, out_features, (batch_size,), device=device)
        freqs = torch.where(targets == 0, 1.0, 4.5).view(1, batch_size, 1)
        phase_grad = torch.where(targets == 0, 0.0, math.pi / 2).view(1, batch_size, 1)
        ch_idx = torch.arange(in_features, device=device).view(1, 1, in_features)

        phases = 2 * math.pi * freqs * t_norm + ch_idx * phase_grad
        inputs = torch.polar(torch.ones_like(phases) * 0.9, phases)

        optimizer.zero_grad()
        logits, spikes = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == targets).float().mean().item() * 100
        print(f"Step {step:02d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}% | Spikes: {spikes.abs().sum().item():.0f}")
`;
