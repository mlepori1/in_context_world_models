"""Tests for a subset of data_utils."""

import numpy as np

import unittest
import numpy as np

from .. import data_utils
from .. import prompt_bank
from .. import rule_bank


class DataUtilsTest(unittest.TestCase):

  ### Test random walk data generation ###
  def test_create_random_walk_data(self):
    grid = np.array([
        ['a', 'b', 'c'],
        ['d', 'e', 'f'],
        ['g', 'h', 'i'],
    ])

    # Very basic test: check that the length of the walk is correct
    walk, _ = data_utils.create_random_walk_data(grid, 3)
    self.assertTrue(len(walk) == 3)

    # Test that start_position is respected
    start_position = (1, 1)
    walk, _ = data_utils.create_random_walk_data(grid, 3, start_position)
    self.assertEqual(walk[0], 'e')

    # Test that the model only produces valid transitions
    grid = np.array([
        ['a', 'b'],
        ['c', 'd'],
    ])
    # In this grid, "a" can never go to "d", "c" can never go to "b",
    # and vice versa
    for _ in range(100):
      walk, _ = data_utils.create_random_walk_data(grid, 1000)
      walk_str = ' '.join(walk)
      self.assertNotIn('a d', walk_str)
      self.assertNotIn('d a', walk_str)
      self.assertNotIn('c b', walk_str)
      self.assertNotIn('b c', walk_str)
      self.assertIn('a b', walk_str)
      self.assertIn('b a', walk_str)
      self.assertIn('a c', walk_str)
      self.assertIn('c a', walk_str)
      self.assertIn('b d', walk_str)
      self.assertIn('d b', walk_str)
      self.assertIn('c d', walk_str)
      self.assertIn('d c', walk_str)

  def test_create_rule_data(self):
    grid = np.array([
        ['a', 'b', 'c'],
        ['d', 'e', 'f'],
        ['g', 'h', 'i'],
    ])
    rule_function = rule_bank.two_step_vertical_rule
    self.assertRaises(
        ValueError, data_utils.create_rule_data, grid, rule_function, 3
    )
    rule_examples, _ = data_utils.create_rule_data(grid, rule_function, -1)
    self.assertTrue(len(rule_examples) == 2)

    # Test that the held out example is never a rule example, and that the held
    # out example is correct
    for _ in range(100):
      rule_examples, held_out_example = data_utils.create_rule_data(
          grid, rule_function, -1
      )
      if held_out_example[0] == 'a':
        for example in rule_examples:
          self.assertNotEqual(example[0], 'a')
        self.assertEqual(held_out_example[1], 'g')
      elif held_out_example[0] == 'b':
        for example in rule_examples:
          self.assertNotEqual(example[0], 'b')
        self.assertEqual(held_out_example[1], 'h')
      elif held_out_example[0] == 'c':
        for example in rule_examples:
          self.assertNotEqual(example[0], 'c')
        self.assertEqual(held_out_example[1], 'i')
      else:
        self.fail('Held out example had unexpected value.')
      self.assertTrue(len(rule_examples) == 2)

  def test_create_metalearning_data(self):
    grid = np.array([
        ['a', 'b', 'c', 'd'],
        ['e', 'f', 'g', 'h'],
        ['i', 'j', 'k', 'l'],
        ['m', 'n', 'o', 'p'],
    ])
    rule_function = rule_bank.one_step_vertical_rule
    num_rule_examples = 9
    walk_length = 5
    num_metalearning_examples = 3
    structure, rule_examples, held_out_example = (
        data_utils.create_metalearning_data(
            grid,
            rule_function,
            num_rule_examples,
            walk_length,
            num_metalearning_examples,
        )
    )
    prompt, _, _ = prompt_bank.metalearning_no_information(
        structure, rule_examples, held_out_example
    )
    print(prompt)
    # Uncomment to see the prompt
    # self.assertFalse(True)

    self.assertTrue(len(structure) == 3)
    self.assertTrue(len(structure[0]) == 5)
    self.assertTrue(len(rule_examples) == 3)
    self.assertTrue(len(rule_examples[0]) == 3)
    self.assertIsInstance(held_out_example, tuple)

  def test_create_1d_data(self):
    grid = np.array([
        ['a', 'b', 'c'],
        ['d', 'e', 'f'],
        ['g', 'h', 'i'],
    ])
    rule_function = rule_bank.one_step_vertical_rule
    walk_length = 3
    num_rule_examples = 4
    (
        structure,
        rule_examples,
        (query_example, next_token_answers),
        held_out_example,
    ) = data_utils.create_1d_data(
        grid, walk_length, rule_function, num_rule_examples
    )
    self.assertTrue(len(structure) == walk_length)
    self.assertTrue(len(rule_examples) == num_rule_examples)
    self.assertIsInstance(held_out_example, tuple)

    prompt, _, _ = prompt_bank.no_information_1d(
        structure, rule_examples, held_out_example
    )
    print(f'No Information ICWM prompt:\n{prompt}')
    print(f'Answer: {held_out_example[1]}')

    prompt, _, _ = prompt_bank.next_token_prediction_1d(
        structure, query_example
    )
    print(f'NTP ICWM prompt:\n{prompt}')
    print(f'Answer: {next_token_answers}')
    # Uncomment to see the prompts
    # self.assertFalse(True)


if __name__ == '__main__':
  unittest.main()
