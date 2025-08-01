"""
Model imports for nnterp.

This module provides centralized imports for transformer models with ArchitectureNotFound fallback.
"""

# Dummy class for missing transformer architectures
class ArchitectureNotFound:
    pass


try:
    from transformers import OPTForCausalLM
except ImportError:
    OPTForCausalLM: type[ArchitectureNotFound | OPTForCausalLM] = ArchitectureNotFound

try:
    from transformers import MixtralForCausalLM
except ImportError:
    MixtralForCausalLM: type[ArchitectureNotFound | MixtralForCausalLM] = ArchitectureNotFound

try:
    from transformers import BloomForCausalLM
except ImportError:
    BloomForCausalLM: type[ArchitectureNotFound | BloomForCausalLM] = ArchitectureNotFound

try:
    from transformers import GPT2LMHeadModel
except ImportError:
    GPT2LMHeadModel: type[ArchitectureNotFound | GPT2LMHeadModel] = ArchitectureNotFound

try:
    from transformers import Qwen2MoeForCausalLM
except ImportError:
    Qwen2MoeForCausalLM: type[ArchitectureNotFound | Qwen2MoeForCausalLM] = ArchitectureNotFound

try:
    from transformers import DbrxForCausalLM
except ImportError:
    DbrxForCausalLM: type[ArchitectureNotFound | DbrxForCausalLM] = ArchitectureNotFound

try:
    from transformers import GPTJForCausalLM
except ImportError:
    GPTJForCausalLM: type[ArchitectureNotFound | GPTJForCausalLM] = ArchitectureNotFound

try:
    from transformers import LlamaForCausalLM
except ImportError:
    LlamaForCausalLM: type[ArchitectureNotFound | LlamaForCausalLM] = ArchitectureNotFound

try:
    from transformers import OlmoeForCausalLM
except ImportError:
    OlmoeForCausalLM: type[ArchitectureNotFound | OlmoeForCausalLM] = ArchitectureNotFound
