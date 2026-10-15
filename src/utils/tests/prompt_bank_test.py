"""Tests for a subset of prompt_bank."""

import numpy as np

import unittest
import numpy as np

from .. import prompt_bank


class PromptBankTest(unittest.TestCase):

  ### Test next token prediction prompts ###
  def test_next_token_prediction(self):
    random_walk = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    prompt, walk_part, _ = prompt_bank.next_token_prediction(random_walk)

    ground_truth = (
        'I will present a sequence of words. Generate the next word in the'
        ' sequence. Generate the token [ANSWER], then'
        ' generate the next word in the sequence.\n'
        '[SEQUENCE] a b c d e f g h i j'
    )

    self.assertEqual(prompt, ground_truth)
    self.assertEqual(walk_part, '[SEQUENCE] a b c d e f g h i j')

  def test_next_token_prediction_1d(self):
    random_walk = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    query_example = 'a'
    prompt, walk_part, _ = prompt_bank.next_token_prediction_1d(
        random_walk, query_example
    )
    ground_truth = (
        'I will present a sequence of words. Generate the next word in the'
        ' sequence. Generate the token [ANSWER], then'
        ' generate the next word in the sequence.\n'
        '[SEQUENCE] a b\n c d\n e f\n a'
    )
    print(prompt)
    self.assertEqual(prompt, ground_truth)
    self.assertEqual(walk_part, '[SEQUENCE] a b\n c d\n e f\n a')

  ### Test next token prediction prompts ###
  def test_alt_next_token_prediction(self):
    random_walk = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    prompt, walk_part, _ = prompt_bank.alternative_next_token_prediction(
        random_walk
    )

    ground_truth = (
        'Your job is to predict the next word in a sequence of words.'
        ' Generate the token [ANSWER], then'
        ' generate the next word in the sequence.\n'
        '[SEQUENCE] a b c d e f g h i j'
    )

    self.assertEqual(prompt, ground_truth)
    self.assertEqual(walk_part, '[SEQUENCE] a b c d e f g h i j')

  def test_alt_next_token_prediction_1d(self):
    random_walk = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    query_example = 'a'
    prompt, walk_part, _ = prompt_bank.alternative_next_token_prediction_1d(
        random_walk, query_example
    )
    ground_truth = (
        'Your job is to predict the next word in a sequence of words.'
        ' Generate the token [ANSWER], then'
        ' generate the next word in the sequence.\n'
        '[SEQUENCE] a b\n c d\n e f\n a'
    )
    print(prompt)
    self.assertEqual(prompt, ground_truth)
    self.assertEqual(walk_part, '[SEQUENCE] a b\n c d\n e f\n a')

  def test_next_token_prediction_prepopulated(self):
    random_walk = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    prompt, walk_part, _ = prompt_bank.next_token_prediction_prepopulated(
        random_walk
    )
    ground_truth_prompt = 'Continue the sequence of words.'
    ground_truth_walk_part = '[SEQUENCE] a b c d e f g h i j'
    self.assertEqual(prompt, ground_truth_prompt)
    self.assertEqual(walk_part, ground_truth_walk_part)

  def test_1d_next_token_prediction_prepopulated(self):
    random_walk = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    query_example = 'a'
    prompt, walk_part, _ = prompt_bank.next_token_prediction_1d_prepopulated(
        random_walk, query_example
    )
    ground_truth = (
        'Continue the sequence of words.'
    )
    self.assertEqual(prompt, ground_truth)
    self.assertEqual(walk_part, '[SEQUENCE] a b\n c d\n e f\n a')

  # ### Test No Information prompts ###
  def test_no_information(self):
    random_walk = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    rule_examples = [('a', 'b'), ('b', 'c')]
    query_example = ('f', 'g')
    prompt, walk_part, rule_part = prompt_bank.no_information(
        random_walk, rule_examples, query_example
    )
    ground_truth_prompt = (
        '[SEQUENCE] a b c d e f g h i j\n'
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n\n'
        'Answer the following question by producing the final output word.'
        ' Generate the token [ANSWER], then'
        ' generate the answer.\n'
        '[QUESTION]\nInput: f Output:'
    )
    ground_truth_walk_part = '[SEQUENCE] a b c d e f g h i j'
    ground_truth_rule_part = (
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n'
    )
    self.assertEqual(prompt, ground_truth_prompt)
    self.assertEqual(walk_part, ground_truth_walk_part)
    self.assertEqual(rule_part, ground_truth_rule_part)

  def test_no_information_prepopulated(self):
    random_walk = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    rule_examples = [('a', 'b'), ('b', 'c')]
    query_example = ('f', 'g')
    prompt, walk_part, rule_part = prompt_bank.no_information_prepopulated(
        random_walk, rule_examples, query_example
    )
    ground_truth_prompt = (
        'Complete the pattern.'
    )
    ground_truth_walk_part = '[SEQUENCE] a b c d e f g h i j'
    ground_truth_rule_part = (
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n'
        '[EXAMPLE]\nInput: f Output:'
    )
    self.assertEqual(prompt, ground_truth_prompt)
    self.assertEqual(walk_part, ground_truth_walk_part)
    self.assertEqual(rule_part, ground_truth_rule_part)

  def test_no_information_1d(self):
    random_adjacencies = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    rule_examples = [('a', 'b'), ('b', 'c')]
    query_example = ('f', 'g')
    prompt, walk_part, rule_part = prompt_bank.no_information_1d(
        random_adjacencies, rule_examples, query_example
    )
    ground_truth_prompt = (
        '[SEQUENCE] a b\n c d\n e f\n\n'
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n\n'
        'Answer the following question by producing the final output word.'
        ' Generate the token [ANSWER], then generate the answer.\n'
        '[QUESTION]\nInput: f Output:'
    )
    ground_truth_walk_part = '[SEQUENCE] a b\n c d\n e f\n'
    ground_truth_rule_part = (
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n'
    )
    self.assertEqual(prompt, ground_truth_prompt)
    self.assertEqual(walk_part, ground_truth_walk_part)
    self.assertEqual(rule_part, ground_truth_rule_part)

  def test_1d_no_information_prepopulated(self):
    random_adjacencies = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    rule_examples = [('a', 'b'), ('b', 'c')]
    query_example = ('f', 'g')
    prompt, walk_part, rule_part = prompt_bank.no_information_1d_prepopulated(
        random_adjacencies, rule_examples, query_example
    )
    ground_truth_prompt = (
        'Complete the pattern.'
    )
    ground_truth_walk_part = '[SEQUENCE] a b\n c d\n e f\n'
    ground_truth_rule_part = (
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n'
        '[EXAMPLE]\nInput: f Output:'
    )
    self.assertEqual(prompt, ground_truth_prompt)
    self.assertEqual(walk_part, ground_truth_walk_part)
    self.assertEqual(rule_part, ground_truth_rule_part)

  ### Test Explicit Grid prompts ###
  def test_explicit_grid(self):
    grid = np.array([
        ['a', 'b', 'c'],
        ['d', 'e', 'f'],
        ['g', 'h', 'i'],
    ])
    rule_examples = [('a', 'b'), ('b', 'c'),]
    query_example = ('f', 'g')
    prompt, _, _ = prompt_bank.explicit_grid(
        grid, rule_examples, query_example
    )
    ground_truth_prompt = (
        '[GRID]\n'
        'Coordinates: 0 0 Item: a\n'
        'Coordinates: 0 1 Item: b\n'
        'Coordinates: 0 2 Item: c\n'
        'Coordinates: 1 0 Item: d\n'
        'Coordinates: 1 1 Item: e\n'
        'Coordinates: 1 2 Item: f\n'
        'Coordinates: 2 0 Item: g\n'
        'Coordinates: 2 1 Item: h\n'
        'Coordinates: 2 2 Item: i\n\n'
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n'
        'Answer the following question by producing the final output state.'
        ' Generate the token [ANSWER], then generate the answer.\n'
        '[QUESTION]\nInput: f Output:'
    )
    self.assertEqual(prompt, ground_truth_prompt)

  def test_explicit_grid_prepopulated(self):
    grid = np.array([
        ['a', 'b', 'c'],
        ['d', 'e', 'f'],
        ['g', 'h', 'i'],
    ])
    rule_examples = [('a', 'b'), ('b', 'c')]
    query_example = ('f', 'g')
    prompt, grid_part, rule_part = prompt_bank.explicit_grid_prepopulated(
        grid, rule_examples, query_example
    )
    ground_truth_grid_part = (
        '[GRID]\n'
        'Coordinates: 0 0 Item: a\n'
        'Coordinates: 0 1 Item: b\n'
        'Coordinates: 0 2 Item: c\n'
        'Coordinates: 1 0 Item: d\n'
        'Coordinates: 1 1 Item: e\n'
        'Coordinates: 1 2 Item: f\n'
        'Coordinates: 2 0 Item: g\n'
        'Coordinates: 2 1 Item: h\n'
        'Coordinates: 2 2 Item: i\n'
    )
    ground_truth_rule_part = (
        '\n[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n'
        '[EXAMPLE]\nInput: f Output:'
    )
    ground_truth_prompt = 'Complete the pattern.'
    self.assertEqual(prompt, ground_truth_prompt)
    self.assertEqual(grid_part, ground_truth_grid_part)
    self.assertEqual(rule_part, ground_truth_rule_part)

  ### Test Metalearning prompts ###
  def test_metalearning_no_information(self):
    random_walk = [
        ['a', 'b', 'c', 'd'],
        ['e', 'f', 'g', 'h'],
    ]
    rule_examples = [
        [('a', 'b'), ('b', 'c')],
        [('a', 'b'), ('e', 'f')],
    ]
    query_example = ('f', 'g')
    prompt, _, _ = prompt_bank.metalearning_no_information(
        random_walk, rule_examples, query_example
    )
    ground_truth_prompt = (
        '[SEQUENCE] a b c d\n'
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n\n'
        '[SEQUENCE] e f g h\n'
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: e Output: f\n\n'
        'Answer the following question by producing the final output word.'
        ' Generate the token [ANSWER], then generate the answer.\n'
        '[QUESTION]\nInput: f Output:'
    )
    self.assertEqual(prompt, ground_truth_prompt)

  def test_metalearning_no_information_prepopulated(self):
    random_walk = [
        ['a', 'b', 'c', 'd'],
        ['e', 'f', 'g', 'h'],
    ]
    rule_examples = [
        [('a', 'b'), ('b', 'c')],
        [('a', 'b'), ('e', 'f')],
    ]
    query_example = ('f', 'g')
    prompt, prepopulated_part, _ = (
        prompt_bank.metalearning_no_information_prepopulated(
            random_walk, rule_examples, query_example
        )
    )
    ground_truth_prompt = 'Complete the pattern.'
    ground_truth_prepopulated_part = (
        '[SEQUENCE] a b c d\n'
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: b Output: c\n\n'
        '[SEQUENCE] e f g h\n'
        '[EXAMPLE]\nInput: a Output: b\n'
        '[EXAMPLE]\nInput: e Output: f\n\n'
        '[EXAMPLE]\nInput: f Output:'
    )
    self.assertEqual(prompt, ground_truth_prompt)
    self.assertEqual(prepopulated_part, ground_truth_prepopulated_part)


if __name__ == '__main__':
  unittest.main()
