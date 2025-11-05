# Documentation Editing Skill

You are helping edit nnterp's Sphinx documentation. Follow these critical guidelines:

## Before Starting

Repeat the documentation mantra:
```
I must respect nnsight execution order in all examples.
I must be factual and concise, avoiding verbose language.
I must position nnterp as a nnsight wrapper for transformers.
I will add metadata for llms.txt generation.
I will show tensor shapes where helpful.
I will link liberally to related sections.
```

## Critical Rules

### 1. NNsight Execution Order (MOST IMPORTANT)
All code examples MUST access components in forward pass order:
- ✅ `layers_output[1]` before `layers_output[2]`
- ✅ `attentions_output[i]` before `layers_output[i]`
- ✅ `attentions_input[i]` before `attentions_output[i]`
- ❌ NEVER access later layers before earlier ones
- ❌ NEVER access layer output before its input/components

### 2. Metadata for llms.txt
Every RST file MUST include after the title:
```rst
Title Here
==========

.. meta::
   :llm-description: Brief description for LLM agents (one sentence)

Content starts here...
```

### 3. Tone and Style
- Use demo.py tone: factual and concise
- Avoid verbose language like "core philosophy", "research-first design"
- Position nnterp as "a nnsight wrapper for transformer models"
- NOT as a general mechanistic interpretability tool or nnsight replacement

### 4. Code Examples
- Include tensor shape comments where helpful: `# (batch, seq_len, hidden_dim)`
- All examples must be runnable (or clearly marked as pseudocode)
- Show both correct and incorrect patterns in best practices docs
- Emphasize when features require specific flags (e.g., `enable_attention_probs=True`)

### 5. Cross-Linking
Use `:doc:` directives liberally to link related sections:
```rst
See :doc:`basic-usage` for more details.
```

### 6. Content Organization
- Progressive complexity: quickstart → basic → advanced → reference
- Separate "what works" vs "what's guaranteed"
- Use real examples (not just generic patterns)

## Documentation Structure

### Entry Point (index.rst)
- Positions nnterp as nnsight wrapper for transformers
- Emphasizes users need to know NNsight
- 3 main sections: Getting Started, User Guide, API Reference

### Getting Started
- **quickstart.rst**: 5-minute intro, installation, basic patterns

### User Guide (6 pages)
1. **basic-usage.rst**: Standardized interface, module tree, loading models
2. **interventions.rst**: Analysis methods (logit lens, patchscope)
3. **model-validation.rst**: Testing system, guarantees, trade-offs
4. **adding-model-support.rst**: RenameConfig, custom architectures
5. **advanced-features.rst**: Attention probs, prompt utils, visualization
6. **nnsight-best-practices.rst**: Execution order, gradients, performance
7. **nnsight-utils.rst**: Standalone utilities for raw nnsight models

### API Reference
- **api.rst**: Auto-generated from docstrings

## When Editing

1. Read the target file first
2. Check if metadata directive exists, add if missing
3. Verify all code examples respect execution order
4. Ensure tone matches demo.py (factual, concise)
5. Add tensor shape comments where helpful
6. Add cross-references to related docs
7. Emphasize feature requirements (flags, versions, etc.)

## Common Mistakes to Avoid

❌ Accessing layers out of order
❌ Verbose language ("core philosophy", "powerful framework")
❌ Missing metadata directive
❌ Missing tensor shape comments in complex examples
❌ Positioning nnterp as general MI tool
❌ Code examples that can't run

## After Editing

Ask yourself:
- Does every code example respect nnsight execution order?
- Is the tone factual and concise?
- Is the metadata directive present?
- Are tensor shapes shown where helpful?
- Are related sections cross-referenced?
- Can the code examples run?
