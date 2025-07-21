from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch as th
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .prompt_utils import Prompt


def plot_topk_tokens(
    next_token_probs: th.Tensor,
    tokenizer,
    k: int = 4,
    title: str | None = None,
    use_token_ids: bool = False,
    file: str | Path | None = None,
    save_html: bool = True,
    height: int = 300,
    width: int = 400,
) -> go.Figure:
    """
    Plot the top k tokens for each layer using Plotly.

    Args:
        next_token_probs (th.Tensor): Probability tensor of shape (batch_size, num_layers, vocab_size) or (num_layers, vocab_size) or (vocab_size,)
        tokenizer: Tokenizer object
        k (int): Number of top tokens to plot
        title (str): Title of the plot
        use_token_ids (bool): If True, use token IDs instead of token strings
        file (str, optional): File path to save the plot
        save_html (bool): If True, save an HTML file along with the image

    Returns:
        go.Figure: Plotly figure object
    """

    # Ensure next_token_probs has the correct shape
    if next_token_probs.dim() == 1:
        next_token_probs = next_token_probs.unsqueeze(0).unsqueeze(0)
    elif next_token_probs.dim() == 2:
        next_token_probs = next_token_probs.unsqueeze(0)

    batch_size, num_layers, vocab_size = next_token_probs.shape

    def get_top_tokens(probs: th.Tensor) -> tuple:
        top_tokens = th.topk(probs, k=k, dim=-1)
        top_probs = top_tokens.values.tolist()
        top_token_ids = [[str(t.item()) for t in layer] for layer in top_tokens.indices]
        top_token_strings = [
            ["'" + tokenizer.convert_ids_to_tokens(t.item()) + "'" for t in layer]
            for layer in top_tokens.indices
        ]
        hover_text = [
            [
                f"ID: {id}<br>Token: {token}"
                for id, token in zip(layer_ids, layer_tokens)
            ]
            for layer_ids, layer_tokens in zip(top_token_ids, top_token_strings)
        ]
        return top_probs, top_token_strings, top_token_ids, hover_text

    # Calculate layout
    n_cols = min(3, batch_size)  # Max 3 columns
    n_rows = math.ceil(batch_size / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    for batch_idx, batch_probs in enumerate(next_token_probs):
        top_probs, top_token_strings, top_token_ids, hover_text = get_top_tokens(
            batch_probs
        )
        row = batch_idx // n_cols + 1
        col = batch_idx % n_cols + 1

        heatmap = go.Heatmap(
            z=top_probs,
            x=list(range(k)),
            y=list(range(num_layers)),
            text=top_token_ids if use_token_ids else top_token_strings,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            colorbar=dict(title="Probability", thickness=15, len=0.9),
            hovertext=hover_text,
            hovertemplate="Layer: %{y}<br>%{hovertext}<br>Probability: %{z}<extra></extra>",
        )
        fig.add_trace(heatmap, row=row, col=col)
        fig.update_traces(zmin=0, zmax=1)
        fig.update_xaxes(title_text="Tokens", row=row, col=col)
        fig.update_yaxes(title_text="Layers", row=row, col=col)

    fig.update_layout(
        title=title or f"Top {k} Tokens Heatmap",
        height=height * n_rows,  # Adjust height based on number of rows
        width=width * n_cols,  # Adjust width based on number of columns
        showlegend=False,
    )

    if file:
        if isinstance(file, str):
            file = Path(file)
        if file.suffix != ".html":
            fig.write_image(file, scale=3)
        if save_html or file.suffix == ".html":
            fig.write_html(
                file if file.suffix == ".html" else file.with_suffix(".html")
            )
    fig.show()
    return fig


def prompts_to_df(prompts: list[Prompt], tokenizer=None):
    """
    Convert a list of prompts to a pandas DataFrame, visualizing the target tokens and strings.
    """
    dic = {}
    for i, prompt in enumerate(prompts):
        dic[i] = {"prompt": prompt.prompt}
        for tgt, string in prompt.target_strings.items():
            dic[i][tgt + "_string"] = string
        if tokenizer is not None:
            for tgt, tokens in prompt.target_tokens.items():
                dic[i][tgt + "_tokens"] = tokenizer.convert_ids_to_tokens(tokens)
    return pd.DataFrame.from_dict(dic)
