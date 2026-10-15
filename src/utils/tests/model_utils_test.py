"""Tests for a subset of model_utils."""

import unittest
import functools
import os
import torch
import transformers

from .. import model_utils

TESTMODEL_DIR = 'resources/TinyLlama-1.1B-Chat-v1.0'


class ModelUtilsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = os.path.join(os.path.dirname(__file__), TESTMODEL_DIR)
    cfg = transformers.AutoConfig.from_pretrained(model_path)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 16
    cfg.intermediate_size = 64
    cfg.max_position_embeddings = 200
    self.model = transformers.AutoModelForCausalLM.from_config(cfg)
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_fast=True
    )
    self.tokenizer.add_special_tokens(
        {'additional_special_tokens': [' a', ' b', ' c', ' d']}
    )
    self.model.resize_token_embeddings(len(self.tokenizer))

  def test_find_all_occurrences(self):
    lst = [0, 0, 0, 1, 2, 1, 2]
    self.assertEqual(
        model_utils._find_all_occurrences(lst, 0),
        [0, 1, 2],
    )
    self.assertEqual(
        model_utils._find_all_occurrences(lst, 1),
        [3, 5],
    )
    self.assertEqual(
        model_utils._find_all_occurrences(lst, 2),
        [4, 6],
    )
    self.assertEqual(
        model_utils._find_all_occurrences(lst, 3),
        [],
    )

  def test_find_subsequence_index(self):
    prompt_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4])
    self.assertEqual(
        model_utils._find_subsequence_index(prompt_ids, [0, 1, 2, 3, 4]),
        4,
    )
    self.assertRaises(
        ValueError,
        functools.partial(
            model_utils._find_subsequence_index,
            prompt_ids,
            [0, 1, 2, 3, 4, 5],
        ),
    )

  def test_get_activations(self):
    prompt = ' a b a c d a'
    walk_part = ' a b a'
    rule_part = ' c d a'
    tokens = ['a', 'b', 'c', 'd']
    has_rules = True
    layers = [0, 1]
    prefill_string = None
    token_window_size = 3

    layers2walk, layers2rule = model_utils.get_activations(
        self.model,
        self.tokenizer,
        prompt,
        walk_part,
        rule_part,
        tokens,
        has_rules,
        layers,
        prefill_string,
        token_window_size,
    )

    self.assertTrue(len(layers2walk) == 2)
    self.assertTrue(len(layers2rule) == 2)
    self.assertEqual(layers2walk[0][' a']['num_instances'], 2)
    self.assertEqual(layers2walk[0][' b']['num_instances'], 1)
    self.assertEqual(layers2walk[0][' c']['num_instances'], 0)
    self.assertEqual(layers2walk[0][' d']['num_instances'], 0)

    self.assertEqual(layers2rule[0][' a']['num_instances'], 1)
    self.assertEqual(layers2rule[0][' b']['num_instances'], 0)
    self.assertEqual(layers2rule[0][' c']['num_instances'], 1)
    self.assertEqual(layers2rule[0][' d']['num_instances'], 1)

    # Test with prefill_string, should be identical
    prefill_string = prompt
    prompt = 'complete the sequence'
    layers2walk, layers2rule = model_utils.get_activations(
        self.model,
        self.tokenizer,
        prompt,
        walk_part,
        rule_part,
        tokens,
        has_rules,
        layers,
        prefill_string,
        token_window_size,
    )

    self.assertTrue(len(layers2walk) == 2)
    self.assertTrue(len(layers2rule) == 2)
    self.assertEqual(layers2walk[0][' a']['num_instances'], 2)
    self.assertEqual(layers2walk[0][' b']['num_instances'], 1)
    self.assertEqual(layers2walk[0][' c']['num_instances'], 0)
    self.assertEqual(layers2walk[0][' d']['num_instances'], 0)

    self.assertEqual(layers2rule[0][' a']['num_instances'], 1)
    self.assertEqual(layers2rule[0][' b']['num_instances'], 0)
    self.assertEqual(layers2rule[0][' c']['num_instances'], 1)
    self.assertEqual(layers2rule[0][' d']['num_instances'], 1)

  def test_get_sliding_window_activations(self):
    prompt = ' a b a c d a a a b c d d d'
    walk_part = prompt
    tokens = ['a', 'b', 'c', 'd']
    layers = [0, 1]
    prefill_string = None
    token_window_size = 5

    layers2windows2acts = model_utils.get_sliding_window_activations(
        self.model,
        self.tokenizer,
        prompt,
        walk_part,
        tokens,
        layers,
        token_window_size,
        prefill_string,
    )

    self.assertTrue(len(layers2windows2acts) == 2)
    self.assertTrue(len(layers2windows2acts[0]) == 2)  # Should ignore incomplete window at the end
    self.assertEqual(layers2windows2acts[0][0][' a']['num_instances'], 2)
    self.assertEqual(layers2windows2acts[0][0][' b']['num_instances'], 1)
    self.assertEqual(layers2windows2acts[0][0][' c']['num_instances'], 1)
    self.assertEqual(layers2windows2acts[0][0][' d']['num_instances'], 1)

    self.assertEqual(layers2windows2acts[0][1][' a']['num_instances'], 3)
    self.assertEqual(layers2windows2acts[0][1][' b']['num_instances'], 1)
    self.assertEqual(layers2windows2acts[0][1][' c']['num_instances'], 1)
    self.assertEqual(layers2windows2acts[0][1][' d']['num_instances'], 0)


if __name__ == '__main__':
  unittest.main()
