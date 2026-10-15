"""Runs open model experiments specified in a config file."""

from collections.abc import Sequence
import os
import shutil

from absl import app
from absl import flags
from absl import logging

import ml_collections
from ml_collections import config_flags
import numpy as np

import torch

from ..utils import constants
from ..utils import data_utils
from ..utils import model_utils



torch.compiler.set_stance('force_eager')

config_flags.DEFINE_config_file(
    'config', None, 'Local path to a config describing the experiment.'
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = flags.FLAGS.config
  config = ml_collections.FrozenConfigDict(config)
  print(f'Loaded config:\n{config}')

  # Model hyperparameters.
  logging.info('GPU available: %s', torch.cuda.is_available())

  model, tokenizer = model_utils.load_model(config.model_path)
  logging.info('Model Loaded. Device Map: %s', model.hf_device_map)

  # World Parameters
  world_size = config.world_size
  dimension = config.dimension
  grid_shape = config.grid_shape

  # Prompt Parameters
  prompt_fn = constants.PROMPTS[config.prompt_name]
  rule = constants.RULES[config.rule_name]
  has_rule_examples = constants.PROMPTS_TO_RULE_BOOL[config.prompt_name]
  num_rule_examples = config.num_rule_examples
  walk_length = config.walk_length

  # Experiment Parameters
  num_questions = config.num_questions
  activation_window = config.activation_window
  output_root_folder = config.output_root_folder
  if 'get_activations' in config:
    get_activations = config.get_activations
  else:
    get_activations = False

  # Make output folder
  output_folder = os.path.join(
      output_root_folder,
      config.model_name,
      f'world_size_{world_size}',
      f'dimension_{dimension}',
      f'activation_window_{activation_window}',
      f'{config.prompt_name}_{config.rule_name}',
  )
  os.makedirs(output_folder, exist_ok=True)

  results = {
      'prompt': [],
      'answer': [],
      'response': [],
      'structure_activations': [],
      'rule_activations': [],
      'grid': [],
  }

  # Run experiment
  for i in range(num_questions):

    logging.info(
        'Running question %d, device map: %s', i, model.hf_device_map
    )

    # Generate random grid
    items = constants.ITEMS.copy()
    np.random.shuffle(items)
    grid = np.array(items[:world_size]).reshape(*grid_shape)

    if 'metalearning_no_information' in config.prompt_name:
      next_token_answers = None
      structure, rule_examples, held_out_example = (
          data_utils.create_metalearning_data(
              grid,
              rule,
              num_rule_examples,
              walk_length,
              config.num_metalearning_examples,
          )
      )
      query_example = None
    elif (
        '1d' in config.prompt_name
    ):  # 1D "random walks" are sampled adjacencies
      (
          structure,
          rule_examples,
          (query_example, next_token_answers),
          held_out_example,
      ) = data_utils.create_1d_data(grid, walk_length, rule, num_rule_examples)

    else:
      structure, next_token_answers = data_utils.create_random_walk_data(
          grid, walk_length
      )
      rule_examples, held_out_example = data_utils.create_rule_data(
          grid, rule, num_rule_examples
      )
      query_example = None

    if 'explicit_grid' in config.prompt_name:
      # Explicit grid prompts have a grid as input, so we pass the grid directly
      # to the prompt function.
      prompt, structure_prompt, rule_prompt = prompt_fn(
          grid, rule_examples, held_out_example
      )
      answer = ' ' + held_out_example[1].strip()
    elif 'next_token_prediction' in config.prompt_name:
      # Query example is ignored in 2D next token prediction prompts, since
      # the answer is just the next token in the random walk.
      # 1D next token prediction needs an adjacency to predict from.
      prompt, structure_prompt, rule_prompt = prompt_fn(
          structure, query_example
      )
      answer = None
    else:
      prompt, structure_prompt, rule_prompt = prompt_fn(
          structure, rule_examples, held_out_example
      )
      answer = ' ' + held_out_example[1].strip()

    # If prepopulating model response, we set the structure prompt to be
    # the same as the prompt, and the prefill string to the structure prompt.
    if 'prepopulate' in config.prompt_name:
      if 'metalearning' in config.prompt_name:
        prefill_string = structure_prompt
      elif 'no_information' in config.prompt_name:
        prefill_string = structure_prompt + '\n' + rule_prompt
      elif 'next_token_prediction' in config.prompt_name:
        prefill_string = structure_prompt
      elif 'explicit_grid' in config.prompt_name:
        prefill_string = structure_prompt + rule_prompt
      else:
        raise ValueError(f'Unsupported prompt: {config.prompt_name}')
    else:
      prefill_string = None

    if get_activations:
      # Get activations from random walk and from rule examples
      logging.info('Getting activations')
      if rule_prompt:
        rule_prompt = rule_prompt.rstrip()

      if structure_prompt:
        structure_prompt = structure_prompt.replace('[SEQUENCE]', '')
        # Ensure that the walk_part only contains the random walk,
        # and not the prompt instructions.
        assert np.all([
            token in grid.flatten().tolist()
            for token in structure_prompt.split()
        ])

      layers = config.layers
      structure_activations, rule_activations = model_utils.get_activations(
          model,
          tokenizer,
          prompt,
          structure_prompt,
          rule_prompt,
          grid.flatten().tolist(),
          has_rule_examples=has_rule_examples,
          layers=layers,
          prefill_string=prefill_string,
          token_window_size=activation_window,
      )
      logging.info('Done getting activations')
    else:
      structure_activations = None
      rule_activations = None

    # Generate text responses to prompt
    if 'prepopulate' in config.prompt_name:
      max_length = 1  # Just generate the next token.
    else:
      max_length = 512

    response_text = model_utils.generate_text(
        model,
        tokenizer,
        prompt,
        prefill_string=prefill_string,
        max_length=max_length,
    )

    results['prompt'].append(prompt)
    if 'next_token_prediction' in config.prompt_name:
      results['answer'].append(next_token_answers)
    else:
      results['answer'].append(answer)

    results['response'].append(response_text)
    results['structure_activations'].append(structure_activations)
    results['rule_activations'].append(rule_activations)
    results['grid'].append(grid)

  # Save results
  structure_activations = results['structure_activations']
  rule_activations = results['rule_activations']
  del results['structure_activations']
  del results['rule_activations']

  torch.save(results, './results.pth')
  torch.save(structure_activations, './structure_activations.pth')
  torch.save(rule_activations, './rule_activations.pth')

  if os.path.exists(os.path.join(output_folder, 'results.pth')):
    os.remove(os.path.join(output_folder, 'results.pth'))
  shutil.copy(
      './results.pth',
      os.path.join(output_folder, 'results.pth'),
  )

  if os.path.exists(os.path.join(output_folder, 'structure_activations.pth')):
    os.remove(os.path.join(output_folder, 'structure_activations.pth'))
  shutil.copy(
      './structure_activations.pth',
      os.path.join(output_folder, 'structure_activations.pth'),
  )

  if os.path.exists(os.path.join(output_folder, 'rule_activations.pth')):
    os.remove(os.path.join(output_folder, 'rule_activations.pth'))
  shutil.copy(
      './rule_activations.pth',
      os.path.join(output_folder, 'rule_activations.pth'),
  )

  config_path = config_flags.get_config_filename(flags.FLAGS['config'])
  if os.path.exists(os.path.join(output_folder, os.path.basename(config_path))):
    os.remove(os.path.join(output_folder, os.path.basename(config_path)))
  shutil.copy(
      config_path,
      os.path.join(output_folder, os.path.basename(config_path)),
  )


if __name__ == '__main__':
  app.run(main)
