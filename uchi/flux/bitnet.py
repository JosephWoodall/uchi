import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        # 1.58-bit quantization: {-1, 0, 1}
        gamma = weight.abs().mean()
        # Scale weight by gamma and round
        quantized = torch.round(weight / (gamma + 1e-8))
        # Clip to [-1, 1]
        quantized = torch.clamp(quantized, -1.0, 1.0)
        # Rescale
        return quantized * gamma

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        return grad_output

class ActivationQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 8-bit activation quantization: [-128, 127]
        gamma = x.abs().max(dim=-1, keepdim=True).values
        # Scale to 8-bit range
        scale = 127.0 / (gamma + 1e-8)
        quantized = torch.round(x * scale)
        quantized = torch.clamp(quantized, -128.0, 127.0)
        # Rescale back
        return quantized / scale

    @staticmethod
    def backward(ctx, grad_output):
        # STE
        return grad_output

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.layer_norm = nn.LayerNorm(in_features)
        # Quantization can be disabled for full-precision training (Phase 1).
        # Enable for QAT (Phase 2) or inference.
        self.quantize = True

    def forward(self, x):
        # Apply layer norm
        x_norm = self.layer_norm(x)
        
        if self.quantize:
            # Quantize activations to 8-bit
            x_quant = ActivationQuantizer.apply(x_norm)
            
            # Quantize weights to 1.58-bit (ternary)
            w_quant = WeightQuantizer.apply(self.weight)
            
            # Forward pass with quantized values
            return F.linear(x_quant, w_quant, self.bias)
        else:
            # Full-precision forward pass (better gradient signal during training)
            return F.linear(x_norm, self.weight, self.bias)
