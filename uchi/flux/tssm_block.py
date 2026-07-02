import torch
import torch.nn as nn
from .ssm import SSMCell
from .bitnet import BitLinear

class TSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=32):
        super().__init__()
        self.d_model = d_model
        
        # In projection (using BitLinear)
        self.in_proj = BitLinear(d_model, d_model * 2, bias=False)
        
        # State Space Model cell
        self.ssm = SSMCell(d_model, d_state)
        
        # Out projection (using BitLinear)
        self.out_proj = BitLinear(d_model, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # Linear projection to expand dimension
        x_proj = self.in_proj(x)
        
        # Split into x and residual connection (like gated Mamba)
        x_ssm, res = x_proj.chunk(2, dim=-1)
        
        # Pass through SSM
        y_ssm, final_state = self.ssm(x_ssm)
        
        # Activation and gating
        y = y_ssm * nn.functional.silu(res)
        
        # Output projection
        out = self.out_proj(y)
        
        # Residual connection
        return out + x, final_state
