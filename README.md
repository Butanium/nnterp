# nnterp
## Overview
This might become a real package at some point, for now it's just a package in which I dump my nnsight code.
- `nnsight_utils.py` basically allows you to deal with TL and HF models in a similar way.
- `interventions.py` is a module that contains tools like logit lens, patchscope lens and other interventions.
- `prompt_utils.py` contains utils to create prompts for which you want to track specific tokens in the next token distribution and run interventions on them and collect the probabilities of the tokens you're interested in.

## Installation
- `pip install nnterp`

# Contributing
- Create a git tag with the version number `git tag vx.y.z; git push origin vx.y.z`
- Build with `python -m build`
- Publish with e.g. `twine upload dist/*x.y.z*`