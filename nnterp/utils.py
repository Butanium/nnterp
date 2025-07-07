import sys


def is_notebook():
    """Detect the current Python environment"""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True
        else:
            return False
    except Exception:
        return False


def display_markdown(text: str):
    from IPython.display import display, Markdown

    display(Markdown(text))
