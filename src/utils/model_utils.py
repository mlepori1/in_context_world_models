"""This module contains functions for loading and interacting with pretrained models."""

import logging
import torch
import transformers


def load_model(model_path):
  model = transformers.AutoModelForCausalLM.from_pretrained(
      model_path, device_map='auto', torch_dtype=torch.bfloat16
  )
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_path, use_fast=True
  )
  return model, tokenizer


def _build_message(model: str, msg: str, role: str = 'user'):
  if model == 'gemma':
    return {'role': role, 'content': [{'type': 'text', 'text': msg}]}
  elif model in ['olmo', 'llama']:
    return {'role': role, 'content': msg}
  else:
    raise ValueError(f'Unsupported model: {model}')


def tokenize(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[str],
    prefill_string: str | None = None,
) -> transformers.tokenization_utils_base.BatchEncoding:
  """Tokenizes a batch of messages using the provided tokenizer.

  Args:
    model: The pretrained model.
    tokenizer: The tokenizer to use.
    messages: A single message string or a list of message strings.
    prefill_string: A string to prefill the model response with. If None, no
      prefill string is used.

  Returns:
    A BatchEncoding containing the tokenized inputs.
  """
  if 'gemma' in model.config.name_or_path.lower():
    name = 'gemma'
  elif 'olmo' in model.config.name_or_path.lower():
    name = 'olmo'
  elif 'llama' in model.config.name_or_path.lower():
    name = 'llama'
  else:
    raise ValueError(f'Unsupported model: {model.config.name_or_path}')

  messages = [_build_message(name, msg) for msg in messages]
  if prefill_string is not None:
    messages.append(_build_message(name, prefill_string, role='assistant'))
    formatted_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
  else:
    formatted_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
  return tokenizer(
      formatted_messages, return_tensors='pt', add_special_tokens=False
  )


def _find_all_occurrences(lst, item):
  """Returns the indices of all occurrences of an item in a list."""
  indices = []
  for i in range(len(lst)):
    if lst[i] == item:
      indices.append(i)
  return indices


def _find_subsequence_index(
    prompt_ids: torch.Tensor, sub_ids: list[int]) -> int:
  """Finds the first starting index of a sub-sequence of token IDs within a prompt tensor.

  Args:
      prompt_ids: The main sequence of token IDs (e.g., the input prompt).
      sub_ids: The short sequence of token IDs to find.

  Returns:
      The first index where the sub-sequence starts, or raise an error if not
      found.
  """
  # 1. Convert PyTorch Tensors to Python lists
  prompt_list = prompt_ids.tolist()

  # 2. Iterate through the main list to find the start of the sub-sequence
  sub_len = len(sub_ids)

  for i in range(len(prompt_list) - sub_len + 1):
    if prompt_list[i : i + sub_len] == sub_ids:
      return i
  raise ValueError('Subsequence not found in the prompt.')


def _collect_activations(
    activations,
    tokens,
    tokenizer,
    prompt_enc,
    start_index,
    end_index,
    layer,
):
  """Collects and averages activations for specified tokens for a given layer.

  Args:
    activations: The hidden states from the model output, typically a tuple of
      tensors, one for each layer.
    tokens: A list of token strings for which to collect activations.
    tokenizer: The tokenizer used by the model.
    prompt_enc: The tokenized input prompt (BatchEncoding).
    start_index: The starting index in the tokenized prompt from which to
      consider tokens.
    end_index: The ending index in the tokenized prompt from which to consider
      tokens.
    layer: The layer to collect activations for.

  Returns:
    A dictionary which maps each token in `tokens` to its mean activation
    vector across all occurrences within the specified segment of the prompt.
  """
  mean_token_activations = {}

  for curr_layer, activations_for_layer in enumerate(activations):
    if curr_layer != layer:  # Only collect activations for specified layer
      continue

    for token in tokens:
      token = ' ' + token.strip()
      encoded_token = tokenizer.encode(token, add_special_tokens=False)
      # Ensure that the word is encoded as a single token
      assert len(encoded_token) == 1
      token_id = encoded_token[0]
      context_window = prompt_enc.input_ids.tolist()[0][start_index:end_index]
      indices = _find_all_occurrences(context_window, token_id)

      token_activations = []
      for index in indices:
        index = index + start_index
        # Ensure that indexing is correct
        assert prompt_enc.input_ids.tolist()[0][index] == token_id
        token_activations.append(activations_for_layer[0][index])

      if token_activations:
        token_activations = torch.mean(
            torch.stack(token_activations), dim=0
        ).cpu()
      else:
        token_activations = torch.zeros(
            activations_for_layer[0].shape[-1]
        ).cpu()

      mean_token_activations[token] = {
          'mean_activations': token_activations,
          'num_instances': len(indices),
      }
  return mean_token_activations


def get_activations(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: str,
    walk_part: str,
    rule_part: str,
    tokens: list[str],
    has_rule_examples: bool,
    layers: list[int],
    prefill_string: str | None = None,
    token_window_size=50,
):
  """Gets activations for tokens in the random walk and rule parts of the prompts.

  Args:
    model: The pretrained language model.
    tokenizer: The tokenizer corresponding to the model.
    prompt: Input prompt.
    walk_part: Portion of the prompt that corresponds to the random walk.
    rule_part: Portion of the prompt that corresponds to the rule examples.
    tokens: A list of token strings for which to calculate activations.
    has_rule_examples: Whether the prompt has rule examples.
    layers: The layers to collect activations for.
    prefill_string: A string to prefill the model response with. If None, no
      prefill string is used.
    token_window_size: The number of preceding tokens to consider when
      collecting hidden states.

  Returns:
    A tuple containing:
      - layers2walk_activations: A dictionary where keys are layer indices and
        values are dictionaries. The inner dictionaries map each token in
        `tokens` to a list of averaged hidden states across layers for
        occurrences within the `token_window_size` of the random walk portion
        of the prompt.
      - layers2rule_activations: A dictionary with the same structure as
        `walk_activations`, but only containing activations corresponding to the
        rule examples in the prompt.
  """

  prompt_enc = tokenize(
      model,
      tokenizer,
      [prompt],
      prefill_string=prefill_string,
  )
  with torch.inference_mode():
    out = model(**prompt_enc, output_hidden_states=True)
  activations = out.hidden_states

  # Remove trailing whitespace from the walk and rule parts to avoid errors.
  walk_token_ids = tokenizer.encode(
      walk_part.rstrip(), add_special_tokens=False
  )
  rule_token_ids = tokenizer.encode(
      rule_part.rstrip(), add_special_tokens=False
  )

  # First, collect activations for the random walk prompt
  first_walk_idx = _find_subsequence_index(
      prompt_enc.input_ids[0], walk_token_ids
  )

  layers2walk_activations = {}
  for layer in layers:
    acts = _collect_activations(
        activations,
        tokens,
        tokenizer,
        prompt_enc,
        # Last token_window_size tokens are used to compute activations
        first_walk_idx + len(walk_token_ids) - token_window_size,
        first_walk_idx + len(walk_token_ids),
        layer=layer,
    )
    # Ensure that the correct total number of tokens were found
    num_tokens = sum([acts[key]['num_instances'] for key in acts.keys()])
    # 50 for 2D grid walk
    # 100 for 1D Adjacencies
    if token_window_size == 50:
      assert num_tokens == 50, (
          f'num_tokens: {num_tokens}, should be 50, first_walk_idx:'
          f' {first_walk_idx}'
      )
    elif token_window_size == 150:
      assert num_tokens == 100, (
          f'num_tokens: {num_tokens}, should be 100, first_walk_idx:'
          f' {first_walk_idx}'
      )
    else:
      # Used for testing!
      assert token_window_size in [3, 5]
    layers2walk_activations[layer] = acts

  # Second, collect activations for the rule prompt, if there are rule examples
  if not has_rule_examples:
    return layers2walk_activations, None

  first_rule_idx = _find_subsequence_index(
      prompt_enc.input_ids[0], rule_token_ids
  )

  layers2rule_activations = {}
  for layer in layers:
    acts = _collect_activations(
        activations,
        tokens,
        tokenizer,
        prompt_enc,
        first_rule_idx,
        first_rule_idx + len(rule_token_ids),
        layer=layer,
    )
    # Ensure that the correct total number of tokens were found
    # For prepopulated prompts:
    # 21 for 10 example input/output pairs + 1 held out example
    # 13 for 6 example input/output pairs + 1 held out example
    # For Not Prepopulated Prompts:
    # 20 for 10 example input/output pairs
    # 12 for 6 example input/output pairs
    # For Testing: 3 tokens
    num_tokens = sum([acts[key]['num_instances'] for key in acts.keys()])

    if num_tokens not in [3, 20, 21, 12, 13]:
      raise ValueError(
          f'num_tokens: {num_tokens}, should be 21 (20 if not prepopulated)'
          ' or 13 (12 if not prepopulated), or 3 for testing'
      )

    layers2rule_activations[layer] = acts

  return (
      layers2walk_activations,
      layers2rule_activations,
  )


def get_sliding_window_activations(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: str,
    walk_part: str,
    tokens: list[str],
    layers: list[int],
    token_window_size=50,
    prefill_string: str | None = None,
):
  """Gets activations for tokens in the random walk over sliding windows.

  Args:
    model: The pretrained language model.
    tokenizer: The tokenizer corresponding to the model.
    prompt: Input prompt.
    walk_part: Portion of the prompt that corresponds to the random walk.
    tokens: A list of token strings for which to calculate activations.
    layers: The layers to collect activations for.
    token_window_size: The number of preceding tokens to consider when
      collecting hidden states.
    prefill_string: A string to prefill the model response with. If None, no
      prefill string is used.

  Returns:
    A dictionary mapping layers to window starts to dictionaries.
    Each inner dictionary maps each token in `tokens` to
    a averaged hidden states for occurrences within
    a sliding window of size `token_window_size` over the random walk prompt.
  """
  prompt_enc = tokenize(
      model,
      tokenizer,
      [prompt],
      prefill_string=prefill_string,
  )
  with torch.inference_mode():
    out = model(**prompt_enc, output_hidden_states=True)
  activations = out.hidden_states

  walk_token_ids = tokenizer.encode(walk_part, add_special_tokens=False)

  first_walk_idx = _find_subsequence_index(
      prompt_enc.input_ids[0], walk_token_ids
  )

  if first_walk_idx == -1:
    raise ValueError('Walk prompt not found in the prompt.')

  last_walk_idx = first_walk_idx + len(walk_token_ids)

  if last_walk_idx - first_walk_idx % token_window_size == 0:
    raise ValueError(
        'Walk prompt length is not a multiple of token window size. '
        f'Walk length: {last_walk_idx - first_walk_idx}, '
        f'Token window size: {token_window_size}'
    )

  layers2window2activations = {}
  for layer in layers:
    # Collect activations for each segment of the random walk prompt
    # corresponding to the start of a sliding window
    window2activations = {}
    window_index = 0
    for start_idx in range(
        first_walk_idx, last_walk_idx - 1, token_window_size
    ):
      if start_idx + token_window_size > last_walk_idx:
        logging.warning(
            'Dropping last window since it is not fully contained in the walk'
            ' prompt.'
        )
        break
      window_activations = _collect_activations(
          activations,
          tokens,
          tokenizer,
          prompt_enc,
          start_idx,
          start_idx + token_window_size,
          layer=layer,
      )
      # 50 for 2D grid walk
      # 100 for 1D Adjacencies
      num_tokens = sum([
          window_activations[key]['num_instances']
          for key in window_activations.keys()
      ])
      if token_window_size == 50:
        assert (
            num_tokens == 50
        ), f'num_tokens: {num_tokens}, should be 50, start_idx: {start_idx}'
      elif token_window_size == 150:
        assert (
            num_tokens == 100
        ), f'num_tokens: {num_tokens}, should be 100, start_idx: {start_idx}'
      else:
        # Used for testing!
        assert token_window_size in [3, 5]
      window2activations[window_index] = window_activations
      window_index += 1
    layers2window2activations[layer] = window2activations

  return layers2window2activations


def generate_text(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: str,
    prefill_string: str | None = None,
    max_length: int = 512,
):
  """Generates text from a model using the provided prompt."""
  print(f'device map: {model.hf_device_map}')

  if 'gemma' in model.config.name_or_path.lower():
    name = 'gemma'
  elif 'olmo' in model.config.name_or_path.lower():
    name = 'olmo'
  else:
    raise ValueError(f'Unsupported model: {model.config.name_or_path}')

  messages = [_build_message(name, prompt)]

  if prefill_string is not None:
    messages.append(_build_message(name, prefill_string, role='assistant'))
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
  else:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
    )
  inputs = tokenizer(inputs, return_tensors='pt', add_special_tokens=False).to(
      model.device
  )

  outputs = model.generate(
      **inputs,
      max_new_tokens=max_length,
      do_sample=False,
  )

  response_ids = outputs[0][
      inputs.input_ids.shape[-1] :
  ]  # Get tokens *after* the prompt
  response = tokenizer.decode(response_ids, skip_special_tokens=True)
  return response
