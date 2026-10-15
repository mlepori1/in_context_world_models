"""Defines a bank of rules for in-context world modeling.
"""


def null_rule(*args):
  """Returns a tuple of None."""
  if args:
    print('Args ignored for null rule')
  tup = (None, None)
  return tuple(tup)


def two_step_vertical_rule(position, graph_shape):
  """Returns the next position in a two-step vertical rule."""
  new_position = []
  for dim, _ in enumerate(position):
    if dim == 0:
      addition = 2
    else:
      addition = 0
    if position[dim] + addition >= graph_shape[dim]:
      return None
    new_position.append(position[dim] + addition)
  return tuple(new_position)


def one_step_vertical_rule(position, graph_shape):
  """Returns the next position in a one-step vertical rule."""
  new_position = []
  for dim, _ in enumerate(position):
    if dim == 0:
      addition = 1
    else:
      addition = 0
    if position[dim] + addition >= graph_shape[dim]:
      return None
    new_position.append(position[dim] + addition)
  return tuple(new_position)


def two_step_diagonal_rule(position, graph_shape):
  """Returns the next position in a two-step bottom-right diagonal rule."""
  new_position = []
  for dim, _ in enumerate(position):
    if dim in [0, 1]:
      addition = 1
    else:
      addition = 0
    if position[dim] + addition >= graph_shape[dim]:
      return None
    new_position.append(position[dim] + addition)
  return tuple(new_position)


def three_step_knight_rule(position, graph_shape):
  """Returns the next position in a knight rule (+2 dim 0, +1 dim 1)."""
  new_position = []
  for dim, _ in enumerate(position):
    if dim == 0:
      addition = 2
    elif dim == 1:
      addition = 1
    else:
      addition = 0
    if position[dim] + addition >= graph_shape[dim]:
      return None
    new_position.append(position[dim] + addition)
  return tuple(new_position)


def four_step_diagonal_rule(position, graph_shape):
  """Returns the next position in a four-step bottom-right diagonal rule."""
  new_position = []
  for dim, _ in enumerate(position):
    if dim in [0, 1]:
      addition = 2
    else:
      addition = 0
    if position[dim] + addition >= graph_shape[dim]:
      return None
    new_position.append(position[dim] + addition)
  return tuple(new_position)


RULES = {
    'null_rule': null_rule,
    'one_step_vertical_rule': one_step_vertical_rule,
    'two_step_vertical_rule': two_step_vertical_rule,
    'two_step_diagonal_rule': two_step_diagonal_rule,
    'three_step_knight_rule': three_step_knight_rule,
    'four_step_diagonal_rule': four_step_diagonal_rule,
}
