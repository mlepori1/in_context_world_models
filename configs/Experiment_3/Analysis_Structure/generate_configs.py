"""Helper script to generate configs from existing configs."""

from collections.abc import Sequence
import math
import os

from absl import app


MODELS = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
]

DIMENSION = [2]
WORLD_SIZES = [16, 25]
MODEL2NO_THINKING_BUDGET = {
    'gemini-2.5-flash': 0,
    'gemini-2.5-pro': 128,
}


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  script_dir = os.path.dirname(os.path.abspath(__file__))

  for model in MODELS:
    for dimension in DIMENSION:
      for world_size in WORLD_SIZES:
        for thinking in [True, False]:
          # Thinking comparison only enabled for gemini models
          if not thinking and 'gemini' not in model:
            continue
          elif not thinking and 'gemini' in model:
            thinking_budget = MODEL2NO_THINKING_BUDGET[model]
          else:
            thinking_budget = 5000

          # Get specific parameters for this experiment
          if dimension == 1:
            grid_shape = (world_size,)
          else:
            grid_shape = (
                int(math.sqrt(world_size)),
                int(math.sqrt(world_size)),
            )

          results_path = "ROOT_DIR + '/ICWM/results/'"

          # Load template
          template_path = os.path.join(script_dir, 'template.py')
          template_file = open(template_path, 'r').read()

          # Edit template
          template_file = template_file.replace(
              'config.model_name', f'config.model_name = "{model}"'
          )
          template_file = template_file.replace(
              'config.gemini_thinking_budget',
              f'config.gemini_thinking_budget = {thinking_budget}',
          )
          template_file = template_file.replace(
              'config.world_size', f'config.world_size = {world_size}'
          )
          template_file = template_file.replace(
              'config.dimension', f'config.dimension = {dimension}'
          )
          template_file = template_file.replace(
              'config.grid_shape', f'config.grid_shape = {grid_shape}'
          )
          template_file = template_file.replace(
              'config.output_root_folder',
              f'config.output_root_folder = {results_path}',
          )

          # Create new directory for this model, if it doesn't exist
          thinking_str = 'limited_thinking' if not thinking else 'thinking'
          new_dir = os.path.join(model, thinking_str)
          os.makedirs(os.path.join(script_dir, new_dir), exist_ok=True)

          # Save off experiment config
          new_filename = f'Structure_Analysis_d{dimension}_w{world_size}.py'
          new_path = os.path.join(script_dir, new_dir, new_filename)
          new_file = open(new_path, 'w')
          new_file.write(template_file)


if __name__ == '__main__':
  app.run(main)
