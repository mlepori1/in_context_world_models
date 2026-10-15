"""Tests for a subset of analysis_helpers."""

import unittest
import numpy as np
import torch

from .. import analysis_helpers


class AnalysisHelpersTest(unittest.TestCase):

  ### Test Accuracy functions ###
  def test_compute_icwm_accuracy(self):
    # Simple case, well formatted with no extra tokens
    results = {
        'prompt': ['a', 'b', 'c'],
        'response': ['[ANSWER] test', '[ANSWER] tests', '[ANSWER] wrong'],
        'answer': ['test', 'tests', 'test'],
    }
    accs = analysis_helpers.compute_icwm_accuracy(results)
    self.assertEqual(np.mean(accs), 2 / 3)

    # Extra tokens in response
    results = {
        'prompt': ['a', 'b', 'c'],
        'response': [
            '[ANSWER] the answer is test',
            '[ANSWER] tests',
            '[ANSWER] wrong',
        ],
        'answer': ['test', 'tests', 'test'],
    }
    accs = analysis_helpers.compute_icwm_accuracy(results)
    self.assertEqual(np.mean(accs), 2 / 3)

    # Missing answer tag
    results = {
        'prompt': ['a', 'b', 'c'],
        'response': ['test', '[ANSWER] tests', '[ANSWER] wrong'],
        'answer': ['test', 'tests', 'test'],
    }
    accs = analysis_helpers.compute_icwm_accuracy(results)
    self.assertEqual(np.mean(accs), 2/3)

    # Missing space
    results = {
        'prompt': ['a', 'b', 'c'],
        'response': ['[ANSWER]test', '[ANSWER] tests', '[ANSWER] wrong'],
        'answer': ['test', 'tests', 'test'],
    }
    accs = analysis_helpers.compute_icwm_accuracy(results)
    self.assertEqual(np.mean(accs), 2 / 3)

  def test_compute_ntp_accuracy(self):
    # Not prepopulated
    results = {
        'prompt': ['a', 'b', 'c'],
        'response': ['[ANSWER] test', 'tests', '[ANSWER] wrong'],
        'answer': [['test', 'tests'], ['test', 'tests'], ['test', 'tests']],
    }
    accs = analysis_helpers.compute_icwm_accuracy(results, ntp=True)
    self.assertEqual(np.mean(accs), 2 / 3)

    # Prepopulated
    results = {
        'prompt': ['a', 'b'],
        'response': [
            'tok toktwo test',
            'tok toktwo wrong',
        ],
        'answer': [['test', 'tests'], ['tok', 'toktwo']],
    }
    accs = analysis_helpers.compute_icwm_accuracy(
        results, ntp=True
    )
    self.assertEqual(np.mean(accs), 1 / 2)

  ### Test representation analysis functions ###
  def test_compute_dirichlet_energy(self):
    grid = np.array(
        [
            ['a', 'b', 'c'],
            ['d', 'e', 'f'],
            ['g', 'h', 'i'],
        ]
    )

    # Test a standard, easy to compute case: all activations are equal
    # except for 1.
    equal_dict = {
        'mean_activations': torch.tensor([1.0, 0.0, 0.0]),
        'num_instances': 1,
    }
    unequal_dict = {
        'mean_activations': torch.tensor([0.0, 0.0, 1.0]),
        'num_instances': 1,
    }
    tokens2states = {
        ' a': equal_dict,
        ' b': equal_dict,
        ' c': equal_dict,
        ' d': equal_dict,
        ' e': equal_dict,
        ' f': equal_dict,
        ' g': equal_dict,
        ' h': equal_dict,
        ' i': unequal_dict,
    }
    dirichlet_energy, _ = analysis_helpers.compute_dirichlet_energy(
        grid, tokens2states
    )
    # Dirichlet energy should be 8.0 for this case (squared diff of 2.0 for 2
    # adjacencies, counted from either direction)
    self.assertEqual(dirichlet_energy, 8.0)

  def test_compute_distance_correlation(self):
    grid = np.array([
        ['a', 'b', 'c'],
        ['d', 'e', 'f'],
        ['g', 'h', 'i'],
    ])

    # Make the two perfectly uncorrelated
    def generate_random_dict():
      return {
          'mean_activations': torch.tensor(
              np.random.normal(loc=0.0, scale=1.0, size=3)
          ),
          'num_instances': 1,
      }

    tokens2states = {
        ' a': generate_random_dict(),
        ' b': generate_random_dict(),
        ' c': generate_random_dict(),
        ' d': generate_random_dict(),
        ' e': generate_random_dict(),
        ' f': generate_random_dict(),
        ' g': generate_random_dict(),
        ' h': generate_random_dict(),
        ' i': generate_random_dict(),
    }
    distance_correlation, _, _ = (
        analysis_helpers.compute_distance_correlation(grid, tokens2states)
    )
    self.assertLess(distance_correlation, 0.2)

    # Make the two perfectly correlated
    def generate_dict(coord1, coord2):
      return {
          'mean_activations': torch.tensor(np.array([coord1, coord2])),
          'num_instances': 1,
      }
    tokens2states = {
        ' a': generate_dict(0, 0),
        ' b': generate_dict(1, 0),
        ' c': generate_dict(2, 0),
        ' d': generate_dict(0, 1),
        ' e': generate_dict(1, 1),
        ' f': generate_dict(2, 1),
        ' g': generate_dict(0, 2),
        ' h': generate_dict(1, 2),
        ' i': generate_dict(2, 2),
    }
    distance_correlation, _, _ = (
        analysis_helpers.compute_distance_correlation(grid, tokens2states)
    )
    self.assertGreater(distance_correlation, .9)

if __name__ == '__main__':
  unittest.main()
