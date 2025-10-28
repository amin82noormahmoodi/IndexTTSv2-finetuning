"""
LoRA (Low-Rank Adaptation) implementation for IndexTTS2 fine-tuning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class LoRALayer(nn.Module):
    """LoRA layer implementation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Original layer (frozen)
        self.original_layer = None
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA computation: x @ A @ B * scaling
        lora_out = self.lora_B(self.dropout(self.lora_A(x)))
        lora_out = lora_out * self.scaling
        
        # Add bias if specified
        if self.bias:
            lora_out = lora_out + self.bias_param
        
        return lora_out
    
    def merge_weights(self) -> torch.Tensor:
        """Merge LoRA weights with original weights"""
        if self.original_layer is None:
            raise ValueError("Original layer not set")
        
        # Compute merged weights: W + (B @ A) * scaling
        lora_weights = self.lora_B.weight @ self.lora_A.weight * self.scaling
        merged_weights = self.original_layer.weight + lora_weights
        
        return merged_weights


class LoRAWrapper(nn.Module):
    """Wrapper to add LoRA to existing linear layers"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=bias
        )
        self.lora.original_layer = original_layer
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer output (frozen)
        original_out = self.original_layer(x)
        
        # LoRA output
        lora_out = self.lora(x)
        
        # Combine outputs
        return original_out + lora_out


class LoRATransformerBlock(nn.Module):
    """LoRA adapter for transformer blocks"""
    
    def __init__(
        self,
        original_block: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[list] = None
    ):
        super().__init__()
        self.original_block = original_block
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        if target_modules is None:
            target_modules = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
        
        self.target_modules = target_modules
        self.lora_layers = nn.ModuleDict()
        
        # Add LoRA to target modules
        self._add_lora_layers()
    
    def _add_lora_layers(self):
        """Add LoRA layers to target modules"""
        for name, module in self.original_block.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    lora_wrapper = LoRAWrapper(
                        original_layer=module,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout
                    )
                    self.lora_layers[name] = lora_wrapper
    
    def forward(self, *args, **kwargs):
        # Apply LoRA modifications
        for name, lora_wrapper in self.lora_layers.items():
            # Replace the original module with LoRA wrapper
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = self.original_block
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, child_name, lora_wrapper)
            else:
                setattr(self.original_block, child_name, lora_wrapper)
        
        # Forward through original block
        return self.original_block(*args, **kwargs)


class LoRAManager:
    """Manager for LoRA adapters in IndexTTS2"""
    
    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[list] = None
    ):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        if target_modules is None:
            target_modules = [
                'gpt.h.*.attn.c_attn',
                'gpt.h.*.attn.c_proj', 
                'gpt.h.*.mlp.c_fc',
                'gpt.h.*.mlp.c_proj'
            ]
        
        self.target_modules = target_modules
        self.lora_adapters = {}
        
        # Add LoRA adapters
        self._add_lora_adapters()
    
    def _add_lora_adapters(self):
        """Add LoRA adapters to target modules"""
        for name, module in self.model.named_modules():
            if self._is_target_module(name):
                if isinstance(module, nn.Linear):
                    lora_wrapper = LoRAWrapper(
                        original_layer=module,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout
                    )
                    self.lora_adapters[name] = lora_wrapper
                    
                    # Replace module in parent
                    self._replace_module(name, lora_wrapper)
    
    def _is_target_module(self, name: str) -> bool:
        """Check if module should have LoRA adapter"""
        for pattern in self.target_modules:
            if self._match_pattern(name, pattern):
                return True
        return False
    
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Check if module name matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace module in parent"""
        parts = name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def get_lora_parameters(self) -> list:
        """Get all LoRA parameters for optimizer"""
        lora_params = []
        for adapter in self.lora_adapters.values():
            lora_params.extend(adapter.parameters())
        return lora_params
    
    def get_frozen_parameters(self) -> list:
        """Get all frozen parameters"""
        frozen_params = []
        for name, param in self.model.named_parameters():
            if not any(lora_name in name for lora_name in self.lora_adapters.keys()):
                frozen_params.append(param)
        return frozen_params
    
    def save_lora_weights(self, path: str):
        """Save LoRA weights"""
        lora_state_dict = {}
        for name, adapter in self.lora_adapters.items():
            lora_state_dict[f"{name}.lora_A.weight"] = adapter.lora.lora_A.weight
            lora_state_dict[f"{name}.lora_B.weight"] = adapter.lora.lora_B.weight
            if adapter.lora.bias:
                lora_state_dict[f"{name}.bias_param"] = adapter.lora.bias_param
        
        torch.save(lora_state_dict, path)
        print(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        lora_state_dict = torch.load(path, map_location='cpu')
        
        for name, adapter in self.lora_adapters.items():
            if f"{name}.lora_A.weight" in lora_state_dict:
                adapter.lora.lora_A.weight.data = lora_state_dict[f"{name}.lora_A.weight"]
            if f"{name}.lora_B.weight" in lora_state_dict:
                adapter.lora.lora_B.weight.data = lora_state_dict[f"{name}.lora_B.weight"]
            if f"{name}.bias_param" in lora_state_dict:
                adapter.lora.bias_param.data = lora_state_dict[f"{name}.bias_param"]
        
        print(f"LoRA weights loaded from {path}")


def add_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[list] = None
) -> LoRAManager:
    """Add LoRA adapters to model"""
    return LoRAManager(
        model=model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules
    )


if __name__ == "__main__":
    # Test LoRA implementation
    original_layer = nn.Linear(512, 512)
    lora_wrapper = LoRAWrapper(original_layer, rank=16, alpha=16.0)
    
    x = torch.randn(1, 10, 512)
    output = lora_wrapper(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"LoRA parameters: {sum(p.numel() for p in lora_wrapper.lora.parameters())}")
    print(f"Original parameters: {sum(p.numel() for p in original_layer.parameters())}")
