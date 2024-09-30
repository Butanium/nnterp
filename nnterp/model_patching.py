from nnsight import LanguageModel
from nnsight.envoy import Envoy

def del_attr(model: LanguageModel, attr: str):
    return  # TODO: fix this
    if isinstance(model, Envoy):
        delattr(model, attr)
    else:
        delattr(model._envoy, attr)
        delattr(model._model, attr)


def rename_model_modules(model: LanguageModel):
    """
    Rename model modules to match the structure expected by nnsight utils.
    Delete the replaced attributes after renaming.

    Args:
        model (LanguageModel): The nnsight LanguageModel to be modified.
    """
    # if isinstance(model, LanguageModel):
    #     rename_model_modules(model._model)
    # Set model attribute
    if not hasattr(model, "model"):
        if hasattr(model, "transformer"):
            model.model = model.transformer
            del_attr(model, "transformer")
        elif hasattr(model, "gpt_neox"):
            model.model = model.gpt_neox
            del_attr(model, "gpt_neox")
        else:
            raise AttributeError("Could not find 'transformer' or 'gpt_neox' attribute")

    # Set layers attribute
    if not hasattr(model.model, "layers"):
        if hasattr(model.model, "h"):
            model.model.layers = model.model.h
            del_attr(model.model, "h")
        else:
            raise AttributeError("Could not find 'h' or 'layers' attribute")

    # Set norm attribute
    if not hasattr(model.model, "norm"):
        if hasattr(model.model, "ln_f"):
            model.model.norm = model.model.ln_f
            del_attr(model.model, "ln_f")
        elif hasattr(model.model, "final_layer_norm"):
            model.model.norm = model.model.final_layer_norm
            del_attr(model.model, "final_layer_norm")
        else:
            raise AttributeError(
                "Could not find 'ln_f' or 'final_layer_norm' attribute"
            )

    # Set lm_head attribute
    if not hasattr(model, "lm_head"):
        if hasattr(model, "embed_out"):
            model.lm_head = model.embed_out
            del_attr(model, "embed_out")
        else:
            raise AttributeError("Could not find lm_head or equivalent")

    # Ensure self_attn is accessible
    for layer in model.model.layers:
        if not hasattr(layer, "self_attn"):
            if hasattr(layer, "attn"):
                layer.self_attn = layer.attn
                del_attr(layer, "attn")
            elif hasattr(layer, "attention"):
                layer.self_attn = layer.attention
                del_attr(layer, "attention")
            elif hasattr(layer, "self_attention"):
                layer.self_attn = layer.self_attention
                del_attr(layer, "self_attention")
            else:
                raise AttributeError(
                    f"Could not find self_attn or equivalent in layer: {layer}"
                )
