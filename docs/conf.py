import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'nnterp'
copyright = '2024, Clément Dumas'
author = 'Clément Dumas'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
add_module_names = False 