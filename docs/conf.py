import os
import sys
import time

sys.path.insert(0, os.path.abspath(".."))

project = "nnterp"
copyright = "2025, Clément Dumas"
author = "Clément Dumas"

extensions = [
    "sphinx.ext.autodoc",  # Auto documentation from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Boostrap design components
    "nbsphinx",  # Jupyter notebook support
    "sphinx.ext.viewcode",  # Add source links to the generated HTML files
    "sphinx.ext.extlinks",  # Add external links
]

templates_path = ["_templates"]
fixed_sidebar = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "nnterp"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["source-link-navbar.js"]
html_show_sphinx = False
html_show_sourcelink = True
html_copy_source = True
html_sourcelink_suffix = '.txt'

# Important for GitHub Pages
html_baseurl = "https://butanium.github.io/nnterp/"
html_extra_path = [".nojekyll"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
add_module_names = False
html_context = {
    "default_mode": "dark",
    "version_identifier": str(int(time.time())),
}


html_theme_options = {
    "show_nav_level": 3,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/butanium/nnterp",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_prev_next": False,
    "pygments_dark_style": "monokai",
}

# Hide empty left sidebar on all pages
html_sidebars = {"**": []}
