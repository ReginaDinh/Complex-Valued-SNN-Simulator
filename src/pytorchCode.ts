export const pytorchCodeString = `import torch
import torch.nn as nn
import numpy as np

class ComplexSurrogateHeaviside(torch.autograd.Function):
    """
    Surrogate gradient for Complex-Valued Spiking Neurons (cLIF).
    Approximate gradient for non-differentiable magnitude thresholding:
        |U| >= V_th
    We backpropagate gradients scaled by the surrogate derivative of the magnitude,
    and rotate the gradient direction using the phase of the membrane potential.
    """
    @staticmethod
    def forward(ctx, u, threshold, alpha=2.0):
        # u is a complex tensor
        # threshold is a real scalar
        magnitude = torch.abs(u)
        spikes_fired = (magnitude >= threshold).to(u.dtype)
        
        # Save tensors for backward pass
        ctx.save_for_backward(u, torch.tensor(threshold), torch.tensor(alpha))
        
        # Spike output carries phase: s = e^{i * arg(u)} if fired else 0
        phase = torch.angle(u)
        # Avoid division by zero by adding a small epsilon
        complex_spikes = spikes_fired * torch.exp(1j * phase)
        return complex_spikes

    @staticmethod
    def backward(ctx, grad_output):
        u, threshold_tensor, alpha_tensor = ctx.saved_tensors
        threshold = threshold_tensor.item()
        alpha = alpha_tensor.item()
        
        magnitude = torch.abs(u)
        # Fast Sigmoid surrogate gradient on the magnitude difference
        # f'(x) = 1 / (1 + alpha * |x|)^2
        diff = magnitude - threshold
        surrogate_grad_mag = 1.0 / (1.0 + alpha * torch.abs(diff))**2
        
        # Backpropagate gradient through complex numbers
        # The gradient with respect to the complex potential U is rotated by the phase of U
        phase = torch.angle(u)
        grad_u = grad_output * surrogate_grad_mag * torch.exp(-1j * phase)
        
        return grad_u, None, None


class ComplexLIF(nn.Module):
    """
    State-of-the-Art Complex-Valued Leaky Integrate-and-Fire (cLIF) Neuron Module.
    
    Equations:
        1. Current integration: I[t] = input_current
        2. Membrane dynamics with Complex Leak (beta):
           U[t] = beta * U[t-1] + I[t]
           where beta = leak_mag * e^{i * leak_phase}
        3. Threshold: |U[t]| >= V_th
        4. Output Spike: S[t] = e^{i * arg(U[t])} if fired else 0
        5. Reset (Soft/Subtraction):
           U[t] = U[t] - V_th * e^{i * arg(U[t])} if fired
    """
    def __init__(self, num_features, threshold=1.0, leak_mag=0.9, leak_phase=0.1, reset_mechanism='soft'):
        super().__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        
        # Leak can be parameterized as a complex parameter to allow learnable phase rotations!
        # U(t) = beta * U(t-1)
        self.leak_mag = nn.Parameter(torch.tensor([leak_mag] * num_features, dtype=torch.float32))
        self.leak_phase = nn.Parameter(torch.tensor([leak_phase] * num_features, dtype=torch.float32))
        
        self.u = None  # Complex membrane potential state

    def init_state(self, batch_size, device):
        # Initialize membrane potential as zero in complex plane
        self.u = torch.zeros(batch_size, self.num_features, dtype=torch.complex64, device=device)

    def forward(self, input_current):
        if self.u is None or self.u.shape[0] != input_current.shape[0]:
            self.init_state(input_current.shape[0], input_current.device)
            
        # Constrain decay magnitude to [0, 0.99] for stability
        mag = torch.clamp(self.leak_mag, 0.0, 0.99)
        phase = self.leak_phase
        beta = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        
        # Update membrane potential: U(t) = beta * U(t-1) + I(t)
        self.u = beta * self.u + input_current
        
        # Generate spikes with surrogate gradient
        spikes = ComplexSurrogateHeaviside.apply(self.u, self.threshold)
        
        # Reset membrane potential
        spikes_fired = torch.abs(spikes) > 0
        if self.reset_mechanism == 'soft':
            # Subtraction in the direction of the phase: U = U - V_th * e^{i * theta}
            self.u = self.u - spikes * self.threshold
        else:
            # Hard reset: set to 0
            self.u = torch.where(spikes_fired, torch.zeros_like(self.u), self.u)
            
        return spikes


class ComplexLinear(nn.Module):
    """
    Complex-Valued Linear Layer: Y = W * X + B
    where W, X, B are complex numbers.
    Mathematically:
        Re(Y) = Re(W) * Re(X) - Im(W) * Im(X) + Re(B)
        Im(Y) = Re(W) * Im(X) + Im(W) * Re(X) + Im(B)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initializing real and imaginary parts of weights
        self.fc_real = nn.Linear(in_features, out_features, bias=True)
        self.fc_imag = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        # x is complex tensor (batch_size, in_features)
        x_real = x.real
        x_imag = x.imag
        
        # Matrix multiplication in complex field
        out_real = self.fc_real(x_real) - self.fc_imag(x_imag)
        out_imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        
        return torch.complex(out_real, out_imag)


class ComplexSNN(nn.Module):
    """
    A 3-layer Complex-Valued Spiking Neural Network for Spectral/Phase Pattern Recognition.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = ComplexLinear(in_dim, hidden_dim)
        self.lif1 = ComplexLIF(hidden_dim, threshold=1.0, leak_mag=0.85, leak_phase=0.15)
        
        self.layer2 = ComplexLinear(hidden_dim, out_dim)
        self.lif2 = ComplexLIF(out_dim, threshold=1.0, leak_mag=0.85, leak_phase=0.15)

    def forward(self, x_seq):
        # x_seq size: (time_steps, batch_size, in_dim) - complex tensor
        time_steps, batch_size, _ = x_seq.shape
        
        # Reset LIF states
        self.lif1.init_state(batch_size, x_seq.device)
        self.lif2.init_state(batch_size, x_seq.device)
        
        # Record output spikes over time
        out_spikes_seq = []
        
        for t in range(time_steps):
            x_t = x_seq[t]
            
            # Layer 1
            cur1 = self.layer1(x_t)
            spk1 = self.lif1(cur1)
            
            # Layer 2
            cur2 = self.layer2(spk1)
            spk2 = self.lif2(cur2)
            
            out_spikes_seq.append(spk2)
            
        # Stack output spikes: (time_steps, batch_size, out_dim)
        out_spikes_seq = torch.stack(out_spikes_seq, dim=0)
        
        # We classify based on average spike magnitude (firing rate) across time
        # or aggregate complex phase of spikes
        avg_spike_intensity = torch.mean(torch.abs(out_spikes_seq), dim=0) # (batch_size, out_dim)
        
        return avg_spike_intensity, out_spikes_seq


# ==========================================================
# SIMULATION & TRAINING EXAMPLE (FOR DEMO)
# ==========================================================
if __name__ == "__main__":
    # Generate synthetic complex-valued inputs (e.g., FFT spectral data)
    # 10 batch size, 20 time steps, 16 features
    batch_size = 10
    time_steps = 25
    in_features = 16
    hidden_features = 32
    out_features = 3  # 3 target classes
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate random magnitude [0, 1.5] and phase [-pi, pi]
    mag = torch.rand(time_steps, batch_size, in_features) * 1.5
    phase = (torch.rand(time_steps, batch_size, in_features) - 0.5) * 2 * np.pi
    inputs = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)).to(device)
    
    # Target classes (0, 1, or 2)
    targets = torch.randint(0, out_features, (batch_size,)).to(device)
    
    # Initialize the network
    model = ComplexSNN(in_dim=in_features, hidden_dim=hidden_features, out_dim=out_features).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training step
    optimizer.zero_grad()
    
    # Forward pass: returns average spike intensity of output neurons
    output_intensity, all_spikes = model(inputs)
    
    # Calculate loss
    loss = criterion(output_intensity, targets)
    
    # Backward pass & Optimize
    loss.backward()
    optimizer.step()
    
    print(f"Successfully ran single training step!")
    print(f"Loss: {loss.item():.4f}")
    print(f"Output Intensity Shape: {output_intensity.shape}")
    print(f"Complex Spikes Fired: Total magnitude of output spikes over time is {torch.sum(torch.abs(all_spikes)).item():.2f}")
`;
