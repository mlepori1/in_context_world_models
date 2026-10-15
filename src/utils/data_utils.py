"""This module contains functions for creating datasets for the in-context world model project.
"""

import numpy as np


def _get_random_walk_neighbors(current_position):
  """Returns a list of valid directions for a random walk over a grid."""
  neighbors = []
  # Only change one dimension at a time
  for dim in range(len(current_position)):
    neighbor_plus = []
    neighbor_minus = []
    for other_dim in range(len(current_position)):
      # Keep other dimensions constant
      if dim != other_dim:
        neighbor_plus.append(current_position[other_dim])
        neighbor_minus.append(current_position[other_dim])
      else:
        neighbor_plus.append(current_position[other_dim] + 1)
        neighbor_minus.append(current_position[other_dim] - 1)
    neighbors.append(tuple(neighbor_plus))
    neighbors.append(tuple(neighbor_minus))
  return neighbors


def _filter_neighbors(neighbors, grid_shape):
  """Filters neighbors that are out of bounds."""
  valid_neighbor_idxs = []
  for neighbor_idxs in neighbors:
    assert len(neighbor_idxs) == len(grid_shape)
    in_bounds = True
    for neighbor_idx, grid_dim_max in zip(neighbor_idxs, grid_shape):
      if not (0 <= neighbor_idx < grid_dim_max):
        in_bounds = False
        break
    if in_bounds:
      valid_neighbor_idxs.append(neighbor_idxs)
  return valid_neighbor_idxs


def get_valid_neighbors(position, grid):
  """Returns a list of valid neighbors for a given position."""
  neighbors = _get_random_walk_neighbors(position)
  valid_neighbor_idxs = _filter_neighbors(neighbors, grid.shape)
  valid_neighbors = [grid[pos] for pos in valid_neighbor_idxs]
  return valid_neighbor_idxs, valid_neighbors


def create_random_walk_data(
    grid,
    length=1000,
    start_position=None,
):
  """Create a Random walk data over an N-dimensional grid.
  
  Args:
    grid: The grid of world states.
    length: The length of the random walk.
    start_position: The starting position of the random walk. If None, a random
      starting position is chosen.

  Returns:
    A tuple containing:
      - walk: A list of items (states) visited during the random walk.
      - final_neighbors: A list of valid neighbor items from the final position
        of the walk.
  """

  walk = []

  if start_position is None:
    possible_start_positions = list(np.ndindex(grid.shape))
    choice = np.random.choice(len(possible_start_positions))
    start_position = possible_start_positions[choice]

  start_object = grid[start_position].item()
  position = start_position
  walk.append(start_object)

  while len(walk) < length:
    valid_neighbor_positions, valid_neighbors = (
        get_valid_neighbors(position, grid)
    )
    choice = np.random.choice(len(valid_neighbors))
    next_position = valid_neighbor_positions[choice]

    curr_object = valid_neighbors[choice]
    position = next_position

    walk.append(curr_object.item())

  # Get the final query answers
  _, final_neighbors = get_valid_neighbors(position, grid)

  return walk, final_neighbors


def create_rule_data(grid, rule_function, num_examples):
  """Produce rule data for the in-context world model project.

  Given a rule_function, which is a function mapping indices to indices, produce
  examples of state pairs that abide by this rule, as well as a "query" example,
  which is a held out state and the corresponding state transition
  given by the rule_function

  Args:
    grid: The grid of world states.
    rule_function: A function mapping indices to indices, representing the rule.
    num_examples: The number of rule examples to generate, -1 means all except
      1 held out example.

  Returns:
    A tuple containing:
      - examples: A list of (from_item, to_item) tuples representing rule
        examples.
      - held_out_example: A (from_item, to_item) tuple for the held-out query.
  """

  # First, select a rule instance to hold out
  held_out_indices = None
  while held_out_indices is None:
    possible_indices = list(np.ndindex(grid.shape))
    held_out_choice = np.random.choice(len(possible_indices))
    held_out_indices = possible_indices[held_out_choice]
    if rule_function(held_out_indices, grid.shape) is None:
      held_out_indices = None

  # Select examples to demonstrate the rule
  index_tuples = list(np.ndindex(grid.shape))
  index_tuples.remove(held_out_indices)

  # Filter indices where the rule function is None
  valid_indices = [
      index
      for index in index_tuples
      if rule_function(index, grid.shape) is not None
  ]
  index_tuples = valid_indices

  # Select examples
  if num_examples > len(index_tuples) - 1:
    raise ValueError(
        f"Not enough examples ({len(index_tuples)}) to generate"
        f" {num_examples} rule examples."
    )

  if num_examples == -1:
    num_examples = len(index_tuples)

  examples = []
  choices = np.random.choice(
      np.arange(len(index_tuples)), num_examples, replace=False
  )
  for choice in choices:
    from_indices = index_tuples[choice]
    if from_indices == held_out_indices:
      raise ValueError(
          "Held out example should not be a possible rule example."
      )
    to_indices = rule_function(from_indices, grid.shape)
    from_item = grid[from_indices]
    to_item = grid[to_indices]
    examples.append((from_item, to_item))

  held_out_to_indices = rule_function(held_out_indices, grid.shape)
  held_out_from_item = grid[held_out_indices]
  held_out_to_item = grid[held_out_to_indices]
  held_out_example = (held_out_from_item, held_out_to_item)

  return examples, held_out_example


def create_metalearning_data(
    grid,
    rule_function,
    num_rule_examples,
    walk_length,
    num_metalearning_examples,
):
  """Produce rule data for the metalearning task.
  
  Args:
    grid: The grid of world states.
    rule_function: A function mapping indices to indices, representing the rule.
    num_rule_examples: The number of rule examples to generate, -1 means all
      except 1 held out example.
    walk_length: The length of each random walk.
    num_metalearning_examples: The number of metalearning examples to generate.
    
  Returns:
    A tuple containing:
      - structure: A list of random walks, each of length walk_length.
      - rule_examples: A list of rule examples, each of length num_rule_examples
        / num_metalearning_examples.
      - held_out_example: A (from_item, to_item) tuple for the held-out query.
  """
  structure = []
  for _ in range(num_metalearning_examples):
    curr_structure, _ = (
        create_random_walk_data(
            grid, length=walk_length
        )
    )
    structure.append(curr_structure)

  all_rule_examples, held_out_example = create_rule_data(
      grid, rule_function, num_rule_examples
  )
  num_examples_per_structure = int(
      num_rule_examples / num_metalearning_examples
  )
  rule_examples = []
  for idx in range(num_metalearning_examples):
    rule_examples.append(
        all_rule_examples[
            idx * num_examples_per_structure : (
                idx + 1
            ) * num_examples_per_structure
        ]
    )
  return structure, rule_examples, held_out_example


def create_1d_data(grid, walk_length, rule_function, num_rule_examples):
  """Produce 1D data for the in-context world model project.
  
  Args:
    grid: The grid of world states.
    walk_length: The number of adjacencies.
    rule_function: A function mapping indices to indices, representing the rule.
    num_rule_examples: The number of rule examples to generate, -1 means all
      except 1 held out example.
    
  Returns:
    A tuple containing:
      - structure: A list of adjacencies
      - rule_examples: A list of rule examples.
      - (query_example, next_token_answers): NTP  query and answers.
      - held_out_example: A (from_item, to_item) tuple for the held-out query.
  """
  structure = []
  for _ in range(walk_length):
    adjacency, _ = create_random_walk_data(grid, length=2)
    structure.append(adjacency)

  query_example, next_token_answers = create_random_walk_data(grid, length=1)
  query_example = query_example[0]
  rule_examples, held_out_example = create_rule_data(
      grid, rule_function, num_rule_examples
  )
  return (
      structure,
      rule_examples,
      (query_example, next_token_answers),
      held_out_example,
  )
