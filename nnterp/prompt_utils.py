from typing import Callable
import torch as th
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .nnsight_utils import NNLanguageModel, next_token_probs
from dataclasses import dataclass


def process_tokens_with_tokenization(
    words: str | list[str], tokenizer, i_am_hacky=False
):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        # If you get the value error even with add_prefix_space=False,
        # you can use the following hacky code to get the token without the prefix
        if i_am_hacky:
            hacky_token = tokenizer("🍐", add_special_tokens=False).input_ids
            length = len(hacky_token)
            tokens = tokenizer("🍐" + word, add_special_tokens=False).input_ids
            if tokens[:length] != hacky_token:
                raise ValueError(
                    "I didn't expect this to happen, please check this code"
                )
            if len(tokens) > length:
                final_tokens.append(tokens[length])
        else:
            # Assuming the tokenizer was initialized with add_prefix_space=False
            token = tokenizer(word, add_special_tokens=False).input_ids[0]
            token_with_start_of_word = tokenizer(
                " " + word, add_special_tokens=False
            ).input_ids[0]
            if token == token_with_start_of_word:
                raise ValueError(
                    "Seems like you use a tokenizer that wasn't initialized with add_prefix_space=False. Not good :("
                )
            final_tokens.append(token)
            if (
                token_with_start_of_word
                != tokenizer(" ", add_special_tokens=False).input_ids[0]
            ):
                final_tokens.append(token_with_start_of_word)
    return list(dict.fromkeys(final_tokens))


@dataclass
class Prompt:
    """
    Generic class to represent a prompt with target tokens to track during next token prediction.

    Args:
        prompt: The prompt to use
        target_tokens: A dictionary of target tokens for each target
        target_strings: A dictionary of target strings for each target
    """

    prompt: str
    target_tokens: dict[str, list[int]]
    target_strings: dict[str, str | list[str]] | None = None

    @classmethod
    def from_strings(
        cls,
        prompt: str,
        target_strings: dict[str, str | list[str]] | list[str] | str,
        tokenizer,
    ):
        if isinstance(target_strings, str) or isinstance(target_strings, list):
            target_strings = {"target": target_strings}
        target_tokens = {
            target: process_tokens_with_tokenization(words, tokenizer)
            for target, words in target_strings.items()
        }
        return cls(
            target_tokens=target_tokens,
            target_strings=target_strings,
            prompt=prompt,
        )

    def has_no_collisions(self, ignore_targets: None | str | list[str] = None):
        tokens = self.target_tokens[:]  # Copy the list
        if isinstance(ignore_targets, str):
            ignore_targets = [ignore_targets]
        if ignore_targets is None:
            ignore_targets = []
        for target, target_tokens in self.target_tokens.items():
            if target in ignore_targets:
                continue
            tokens += target_tokens
        return len(tokens) == len(set(tokens))

    def get_target_probs(self, probs, layer=None):
        target_probs = {
            target: probs[:, :, tokens].sum(dim=2).cpu()
            for target, tokens in self.target_tokens.items()
        }
        if layer is not None:
            target_probs = {
                target: probs_[:, layer] for target, probs_ in target_probs.items()
            }
        return target_probs

    @th.no_grad
    def run(self, nn_model, get_probs: Callable):
        """
        Run the prompt through the model and return the probabilities of the next token for both the target tokens.
        """
        probs = get_probs(nn_model, self.prompt)
        return self.get_target_probs(probs)


def next_token_probs_unsqueeze(
    nn_model: NNLanguageModel, prompt: str | list[str], remote=False, **_kwargs
) -> th.Tensor:
    probs = next_token_probs(nn_model, prompt, remote=remote)
    return probs.unsqueeze(1)  # Add a fake layer dimension


@th.no_grad
def run_prompts(
    nn_model: NNLanguageModel,
    prompts: list[Prompt],
    batch_size: int = 32,
    get_probs_func: Callable | None = None,
    func_kwargs: dict | None = None,
    scan: bool = True,
    remote: bool = False,
    tqdm=tqdm,
) -> tuple[th.Tensor, dict[str, th.Tensor]]:
    """
    Run a list of prompts through the model and return the probabilities of the next token for the target tokens.

    Args:
        nn_model: The NNSight model
        prompts: A list of prompts
        batch_size: The batch size to use
        get_probs: The function to get the probabilities of the next token, default to next token prediction
        method_kwargs: The kwargs to pass to the get_probs function
        scan: Whether to use nnsight's scan
        tqdm: The tqdm function to use, default to tqdm.auto.tqdm. Use None to disable tqdm

    Returns:
        A target_probs tensor of shape (num_prompts, num_layers)
    """
    str_prompts = [prompt.prompt for prompt in prompts]
    dataloader = DataLoader(str_prompts, batch_size=batch_size)
    probs = []
    if get_probs_func is None:
        get_probs_func = next_token_probs_unsqueeze
    if func_kwargs is None:
        func_kwargs = {}
    if tqdm is None:
        tqdm = lambda x, **kwargs: x
    for prompt_batch in tqdm(dataloader, total=len(dataloader), desc="Running prompts"):
        probs.append(
            get_probs_func(
                nn_model, prompt_batch, scan=scan, remote=remote, **func_kwargs
            )
        )
        scan = False  # Not sure if this is a good idea
    probs = th.cat(probs)
    target_probs = {target: [] for target in prompts[0].target_tokens.keys()}
    for i, prompt in enumerate(prompts):
        for target, tokens in prompt.target_tokens.items():
            target_probs[target].append(probs[i, :, tokens].sum(dim=1))
    target_probs = {
        target: th.stack(probs).cpu() for target, probs in target_probs.items()
    }
    return target_probs


def prompts_to_df(prompts: list[Prompt], tokenizer=None):
    dic = {}
    for i, prompt in enumerate(prompts):
        dic[i] = {"prompt": prompt.prompt}
        for tgt, string in prompt.target_strings.items():
            dic[i][tgt + "_string"] = string
        if tokenizer is not None:
            for tgt, tokens in prompt.target_tokens.items():
                dic[i][tgt + "_tokens"] = tokenizer.convert_ids_to_tokens(tokens)
    return pd.DataFrame.from_dict(dic)
