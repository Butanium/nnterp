#!/usr/bin/env python3
"""Standalone script to update llms.txt without running full Sphinx build."""

import sys
from pathlib import Path

# Add docs to path so we can import from conf.py
docs_dir = Path(__file__).parent.parent / "docs"
sys.path.insert(0, str(docs_dir))

from conf import generate_llms_txt

if __name__ == "__main__":
    source_dir = docs_dir
    output_file = source_dir / "llms.txt"

    generate_llms_txt(source_dir, output_file)
    print(f"Generated {output_file}")
