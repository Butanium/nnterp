import torch as th
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


def plot_topk_tokens(
    next_token_probs: th.Tensor,
    tokenizer,
    k: int = 4,
    title: str = None,
    use_token_ids: bool = False,
    file: str | None = None,
    save_html: bool = True,
) -> go.Figure:
    """
    Plot the top k tokens for each layer using Plotly.

    Args:
        next_token_probs (th.Tensor): Probability tensor of shape (batch_size, num_layers, vocab_size)
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
        height=300 * n_rows,  # Adjust height based on number of rows
        width=400 * n_cols,  # Adjust width based on number of columns
        showlegend=False,
    )

    if file:
        if not file.endswith(".html"):
            fig.write_image(file, scale=3)
        if save_html or file.endswith(".html"):
            fig.write_html(file.split(".")[0] + ".html")

    return fig
