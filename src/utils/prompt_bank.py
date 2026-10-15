"""Defines prompt functions for the in-context world model project."""

import logging

import numpy as np


#### Next Token Prediction Prompts ####


def next_token_prediction(random_walk: list[str], *args):
  """Builds a prompt to assess next token prediction over random walk."""

  if args:
    logging.warning("Args ignored for next token prediction prompt")

  prefix_part = (
      "I will present a sequence of words. Generate the next word in the"
      " sequence. Generate the token [ANSWER], then"
      " generate the next word in the sequence.\n"
  )

  walk_part = "[SEQUENCE] " + " ".join(random_walk)

  prompt = prefix_part + walk_part

  return prompt, walk_part, None


def alternative_next_token_prediction(random_walk: list[str], *args):
  """Builds a prompt to assess next token prediction over random walk."""

  if args:
    logging.warning("Args ignored for next token prediction prompt")

  prefix_part = (
      "Your job is to predict the next word in a sequence of words."
      " Generate the token [ANSWER], then"
      " generate the next word in the sequence.\n"
  )

  walk_part = "[SEQUENCE] " + " ".join(random_walk)

  prompt = prefix_part + walk_part

  return prompt, walk_part, None


def next_token_prediction_prepopulated(random_walk: list[str], *args):
  """Builds a prompt to assess NTP over random walk when prefixing the response.
  
  Args:
    random_walk: A list of strings representing the random walk.
    *args: Additional arguments, which are ignored.
    
  Returns:
    prompt: The prompt string. This is the user message
    walk_part: The walk part of the prompt. This is prefilled.
    rule_part: There are no rule examples in this prompt, so None.
  """

  if args:
    logging.warning("Args ignored for next token prediction prompt")

  prefix_part = (
      "Continue the sequence of words."
  )

  walk_part = "[SEQUENCE] " + " ".join(random_walk)

  prompt = prefix_part

  return prompt, walk_part, None


def next_token_prediction_1d(
    adjacencies: list[tuple[str, str]],
    query_example: str,
    *args
):
  """Builds an NTP prompt for 1D adjacencies."""

  if args:
    logging.warning("Args ignored for 1D NTP prompt")

  prefix_part = (
      "I will present a sequence of words. Generate the next word in the"
      " sequence. Generate the token [ANSWER], then"
      " generate the next word in the sequence.\n"
  )

  walk_part = "[SEQUENCE]"
  for adjacency in adjacencies:
    walk_part += " " + " ".join(adjacency) + "\n"
  walk_part += " " + query_example

  prompt = prefix_part + walk_part

  return prompt, walk_part, None


def alternative_next_token_prediction_1d(
    adjacencies: list[tuple[str, str]],
    query_example: str,
    *args
):
  """Builds an NTP prompt for 1D adjacencies."""

  if args:
    logging.warning("Args ignored for 1D NTP prompt")

  prefix_part = (
      "Your job is to predict the next word in a sequence of words."
      " Generate the token [ANSWER], then"
      " generate the next word in the sequence.\n"
  )

  walk_part = "[SEQUENCE]"
  for adjacency in adjacencies:
    walk_part += " " + " ".join(adjacency) + "\n"
  walk_part += " " + query_example

  prompt = prefix_part + walk_part

  return prompt, walk_part, None


def next_token_prediction_1d_prepopulated(
    adjacencies: list[tuple[str, str]],
    query_example: str,
    *args
):
  """Builds a prompt to assess NTP over adjacencies when prefixing the response.
  """

  if args:
    logging.warning("Args ignored for 1D NTP prompt")

  prefix_part = (
      "Continue the sequence of words."
  )

  walk_part = "[SEQUENCE]"
  for adjacency in adjacencies:
    walk_part += " " + " ".join(adjacency) + "\n"
  walk_part += " " + query_example

  prompt = prefix_part

  return prompt, walk_part, None


#### Construct Structure Prompts ####


def construct_structure(random_walk: list[str], *args):
  """Builds a prompt to just construct the grid."""

  if args:
    logging.warning("Args ignored for construct structure prompt")

  prefix_part = (
      "There is a set of states and a structured state transition matrix. Each"
      " state is denoted by a completely arbitrary word.\n"
      "First, I will present a random walk over states. Infer the structure"
      " underlying the state transition matrix.\n[RANDOM WALK]"
  )

  walk_part = " " + " ".join(random_walk)

  query_part = (
      "\nFirst, generate the token [STRUCTURE], then describe the structure of"
      " the state transition matrix."
  )

  prompt = prefix_part + walk_part + query_part
  return prompt, walk_part, None


#### Explicit Grid Prompts ####


def _create_explicit_grid_data(grid: np.ndarray):
  """Create string representation of the grid for the explicit grid analysis."""
  data = []

  for dim_tuple in np.ndindex(grid.shape):
    dim_tuple_str = "Coordinates: " + " ".join(
        [str(dim) for dim in list(dim_tuple)]
    )
    obj = grid[dim_tuple]
    dim_tuple_str += f" Item: {obj.item()}"
    data.append(dim_tuple_str)

  return data


def explicit_grid(
    grid: np.ndarray,
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a prompt to assess rule induction over an explicit grid."""

  prefix_part = "[GRID]\n"

  grid_data = _create_explicit_grid_data(grid)
  grid_part = ""
  for entry in grid_data:
    grid_part += entry + "\n"

  example_part = "\n"
  for example in rule_examples:
    example_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )

  query_part = (
      "Answer the following question by producing the final output state."
      " Generate the token [ANSWER], then"
      " generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
  )

  prompt = prefix_part + grid_part + example_part + query_part
  return prompt, None, None


def explicit_grid_prepopulated(
    grid: np.ndarray,
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a prompt to assess rule induction over an explicit grid."""

  prefix_part = "Complete the pattern."

  grid_part = "[GRID]\n"
  grid_data = _create_explicit_grid_data(grid)
  for entry in grid_data:
    grid_part += entry + "\n"

  example_part = "\n"
  for example in rule_examples:
    example_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )
  example_part += f"[EXAMPLE]\nInput: {str(query_example[0])} Output:"

  return prefix_part, grid_part, example_part


#### In-Context World Model Prompts ####


def no_information(
    random_walk: list[str],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a prompt with no additional clues."""

  walk_part = "[SEQUENCE] " + " ".join(random_walk)

  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )

  prompt = (
      f"{walk_part}\n{rule_part}\n"
      "Answer the following question by producing the final output word."
      " Generate the token [ANSWER], then"
      " generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
  )

  return prompt, walk_part, rule_part


def no_information_prepopulated(
    random_walk: list[str],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a prompt with no additional clues."""

  prefix_part = "Complete the pattern."
  walk_part = "[SEQUENCE] " + " ".join(random_walk)

  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )
  rule_part += f"[EXAMPLE]\nInput: {str(query_example[0])} Output:"

  prompt = prefix_part
  return prompt, walk_part, rule_part


def metalearning_no_information(
    random_walk: list[list[str]],
    rule_examples: list[list[tuple[str, str]]],
    query_example: tuple[str, str],
):
  """Builds a prompt with a structure designed for metalearning."""

  assert len(random_walk) == len(rule_examples), (
      "Number of random walks must match number of rule example sets."
  )

  prompt = ""
  for i in range(len(random_walk)):
    walk_part = "[SEQUENCE] " + " ".join(random_walk[i]) + "\n"
    example_part = ""
    for example in rule_examples[i]:
      example_part += (
          f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
      )
    prompt += walk_part + example_part + "\n"

  prompt = (
      f"{prompt}"
      "Answer the following question by producing the final output word."
      " Generate the token [ANSWER], then"
      " generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
  )
  return prompt, None, None


def metalearning_no_information_prepopulated(
    random_walk: list[list[str]],
    rule_examples: list[list[tuple[str, str]]],
    query_example: tuple[str, str],
):
  """Builds a prompt with a structure designed for metalearning."""

  assert len(random_walk) == len(rule_examples), (
      "Number of random walks must match number of rule example sets."
  )

  prefix_part = "Complete the pattern."
  prompt = prefix_part

  prepopulated_part = ""
  for i in range(len(random_walk)):
    walk_part = "[SEQUENCE] " + " ".join(random_walk[i]) + "\n"
    example_part = ""
    for example in rule_examples[i]:
      example_part += (
          f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
      )
    prepopulated_part += walk_part + example_part + "\n"

  prepopulated_part = (
      f"{prepopulated_part}"
      f"[EXAMPLE]\nInput: {str(query_example[0])} Output:"
  )
  return prompt, prepopulated_part, None


def no_information_1d(
    adjacencies: list[tuple[str, str]],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a prompt with no additional clues, presenting random adjacencies."""

  walk_part = "[SEQUENCE]"
  for adjacency in adjacencies:
    walk_part += " " + " ".join(adjacency) + "\n"

  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )

  prompt = (
      f"{walk_part}\n{rule_part}\n"
      "Answer the following question by producing the final output word."
      " Generate the token [ANSWER], then"
      " generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
  )

  return prompt, walk_part, rule_part


def no_information_1d_prepopulated(
    adjacencies: list[tuple[str, str]],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a prompt with no additional clues, presenting random adjacencies."""

  prefix_part = "Complete the pattern."
  walk_part = "[SEQUENCE]"
  for adjacency in adjacencies:
    walk_part += " " + " ".join(adjacency) + "\n"

  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )
  rule_part += f"[EXAMPLE]\nInput: {str(query_example[0])} Output:"

  prompt = prefix_part
  return prompt, walk_part, rule_part

### Prompts with hints for API models ###


def low_information(
    random_walk: list[str],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a low information prompt."""

  prefix_part = (
      "There is a set of states and a structured state transition matrix. Each"
      " state is denoted by a completely arbitrary word.\n"
      "First, I will present a random walk over states. Infer the structure"
      " of the underlying state transition matrix.\n[RANDOM WALK]"
  )

  walk_part = " " + " ".join(random_walk)

  example_prefix = (
      "\nNow, I will present several examples of inputs and outputs to a"
      " function. This function maps states at one position to states at"
      " another.\n"
  )
  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )

  query_part = (
      "Answer the following question by producing the final output state."
      " Generate the token [ANSWER], then"
      " generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
      )

  prompt = prefix_part + walk_part + example_prefix + rule_part + query_part
  return prompt, walk_part, rule_part


def low_information_cot(
    random_walk: list[str],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a low information prompt with chain of thought."""

  prefix_part = (
      "There is a set of states and a structured state transition matrix. Each"
      " state is denoted by a completely arbitrary word.\n"
      "First, I will present a random walk over states. Infer the structure"
      " underlying the state transition matrix.\n[RANDOM WALK]"
  )

  walk_part = " " + " ".join(random_walk)

  example_prefix = (
      "\nNow, I will present several examples of inputs and outputs to a"
      " function. This function maps states at one position to states at"
      " another.\n"
  )

  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )

  query_part = (
      "Answer the following question by producing the final output state."
      " First, generate the token [STRUCTURE], then describe the structure of"
      " the state transition matrix. Next, generate the token [ANSWER], then"
      " generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
  )

  prompt = prefix_part + walk_part + example_prefix + rule_part + query_part

  return prompt, walk_part, rule_part


def high_information(
    random_walk: list[str],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a high information prompt."""

  prefix_part = (
      "There is a set of states organized in a 5x5 grid. Each state is denoted"
      " by a completely arbitrary word.\n"
      "First, I will present a random walk over the grid. Take note of each"
      " state's position in the grid.\n[RANDOM WALK]"
  )

  walk_part = " " + " ".join(random_walk)

  example_prefix = (
      "\nNow, I will present several examples of inputs and outputs to a"
      " function. This function maps states at one position in the grid to"
      " states at another.\n"
  )
  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )

  query_part = (
      "Answer the following question by producing the final output state."
      " Generate the token [ANSWER], then generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
  )

  prompt = prefix_part + walk_part + example_prefix + rule_part + query_part
  return prompt, walk_part, rule_part


def high_information_cot(
    random_walk: list[str],
    rule_examples: list[tuple[str, str]],
    query_example: tuple[str, str],
):
  """Builds a high information prompt with grid-based chain of thought."""

  prefix_part = (
      "There is a set of states organized in a 5x5 grid. Each state is denoted"
      " by a completely arbitrary word.\n"
      "First, I will present a random walk over the grid. Take note of each"
      " state's position in the grid.\n[RANDOM WALK]"
  )

  walk_part = " " + " ".join(random_walk)

  example_prefix = (
      "\nNow, I will present several examples of inputs and outputs to a"
      " function. This function maps states at one position in the grid to"
      " states at another.\n"
  )

  rule_part = ""
  for example in rule_examples:
    rule_part += (
        f"[EXAMPLE]\nInput: {str(example[0])} Output: {str(example[1])}\n"
    )

  query_part = (
      "Answer the following question by producing the final output state."
      " First, generate the token [STRUCTURE], then generate the 5x5 grid with"
      " each state in the correct position. Next, generate the token [ANSWER],"
      " then generate the answer.\n"
      f"[QUESTION]\nInput: {str(query_example[0])} Output:"
  )

  prompt = prefix_part + walk_part + example_prefix + rule_part + query_part
  return prompt, walk_part, rule_part
