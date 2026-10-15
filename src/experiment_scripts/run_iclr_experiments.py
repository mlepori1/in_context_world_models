"""For a given model and structure, collect representation metrics over context.

These metrics are used to choose the appropriate context length for a given
model.
"""

from collections.abc import Sequence
import os
import shutil

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags
import numpy as np
import pandas as pd
import torch

from ..utils import analysis_helpers
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
  assert (
      'next_token_prediction' in config.prompt_name
  ), 'Prompt must be a next token prediction prompt.'
  walk_length = config.walk_length

  # Experiment Parameters
  num_questions = config.num_questions
  activation_window = config.activation_window
  layers = config.layers

  output_root_folder = config.output_root_folder

  # Make output folder
  output_folder = os.path.join(
      output_root_folder,
      config.model_name,
      f'world_size_{world_size}',
      f'dimension_{dimension}',
      f'activation_window_{activation_window}',
      f'{config.prompt_name}',
  )
  os.makedirs(output_folder, exist_ok=True)

  # Create results dictionary
  results = {
      'dirichlet_energy': [],
      'normalized_dirichlet_energy': [],
      'distance_correlation': [],
      'layer': [],
      'window_start': [],
  }

  # Run experiment
  for i in range(num_questions):

    logging.info('Running question %d, device map: %s', i, model.hf_device_map)
    # Generate random grid
    items = constants.ITEMS.copy()
    np.random.shuffle(items)
    grid = np.array(items[:world_size]).reshape(*grid_shape)

    # To reproduce ICLR results, we need to get activations from random walks
    # starting at each state in the grid.
    start_positions = list(np.ndindex(grid.shape))

    # First create the prompts that will be used to collect activations
    prompts = []
    walk_parts = []
    for start_position in start_positions:
      # Generate random walks or adjacencies, and their prompts
      if '1d' in config.prompt_name:
        structure, _, (query_example, _), _ = data_utils.create_1d_data(
            grid, walk_length, constants.RULES['null_rule'], 0
        )
        prompt, walk_part, _ = prompt_fn(structure, query_example)
      else:
        structure, _ = data_utils.create_random_walk_data(
            grid, walk_length, start_position=start_position
        )
        prompt, walk_part, _ = prompt_fn(structure)

      prompts.append(prompt)
      walk_parts.append(walk_part)

    grid_activations = []

    for prompt_idx, prompt in enumerate(prompts):
      logging.info('Running question %d, prompt %d', i, prompt_idx)
      # Get the activations from random walk over sliding windows of size
      # activation_window
      walk_part = walk_parts[prompt_idx]

      if 'prepopulated' in config.prompt_name:
        prefill_string = walk_part
        print(f'prefill_string: {prefill_string}')
        print(f'prompt: {prompt}')
      else:
        prefill_string = None

      if walk_part:
        walk_part = walk_part.replace('[SEQUENCE]', '')

        # Ensure that the walk_part only contains the random walk,
        # and not the prompt instructions.
        assert np.all(
            [token in grid.flatten().tolist() for token in walk_part.split()]
        )

      layers2window2activations = model_utils.get_sliding_window_activations(
          model,
          tokenizer,
          prompt,
          walk_part,
          grid.flatten().tolist(),
          layers,
          token_window_size=activation_window,
          prefill_string=prefill_string,
      )

      grid_activations.append(layers2window2activations)

    # Reformat activations and run analysis
    for layer in layers:
      for window_start in grid_activations[0][layer].keys():
        # Collect activations for all prompts in the same grid,
        # for the same window start.
        all_activations_in_window = [
            layer2window2acts[layer][window_start]
            for layer2window2acts in grid_activations
        ]
        activations_dict = {
            'structure_activations': all_activations_in_window,
            'grid': [grid] * len(all_activations_in_window),
        }
        # Average activations over tokens in a context window,
        # over all random walks that use the same grid.
        processed_activations = (
            analysis_helpers.collect_structure_activations(
                activations_dict, world_size
            )
        )

        # Compute representation metrics, only doing one grid at a time to
        # save memory.
        processed_activations = processed_activations[0]

        dirichlet_energy, normalized_dirichlet_energy = (
            analysis_helpers.compute_dirichlet_energy(
                grid, processed_activations
            )
        )
        results['dirichlet_energy'].append(dirichlet_energy)
        results['normalized_dirichlet_energy'].append(
            normalized_dirichlet_energy
        )
        results['distance_correlation'].append(
            analysis_helpers.compute_distance_correlation(
                grid, processed_activations
            )[0]
        )
        results['layer'].append(layer)
        results['window_start'].append(window_start)

  # Save results
  pd.DataFrame().from_dict(results).to_csv(
      os.path.join('./representation_metrics.csv')
  )
  if os.path.exists(os.path.join(output_folder, 'representation_metrics.csv')):
    os.remove(os.path.join(output_folder, 'representation_metrics.csv'))

  shutil.copy(
      './representation_metrics.csv',
      os.path.join(output_folder, 'representation_metrics.csv'),
  )

if __name__ == '__main__':
  app.run(main)
