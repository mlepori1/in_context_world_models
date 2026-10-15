"""Helper script to generate configs from existing configs."""

from collections.abc import Sequence
import math
import os

from absl import app



MODELS = [
    'gemma-3-4b-it',
    'gemma-3-12b-it',
    'gemma-3-27b-it',
    'olmo-2-13b-it',
]

DIMENSION = [1, 2]
WORLD_SIZES = [16, 25]

RULE = 'null_rule'

Dimension2ActivationWindows = {
    1: 150,
    2: 50,
}

NUM_EXAMPLES = 0


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  script_dir = os.path.dirname(os.path.abspath(__file__))

  for model in MODELS:
    for dimension in DIMENSION:
      for world_size in WORLD_SIZES:
        # Get specific parameters for this experiment
        num_examples = NUM_EXAMPLES
        activation_window = Dimension2ActivationWindows[dimension]

        if dimension == 1:
          grid_shape = (world_size,)
        else:
          grid_shape = (
              int(math.sqrt(world_size)),
              int(math.sqrt(world_size)),
          )

        context_length = 1500
        prompt_name = 'next_token_prediction'
        if dimension == 1:
          prompt_name += '_1d'
        prepopulated_path_str = 'not_prepopulated'
        results_path = "ROOT_DIR + '/ICWM/results_constant_context/'"

        # Load template
        template_path = os.path.join(script_dir, 'template.py')
        template_file = open(template_path, 'r').read()

        # Edit template
        if "gemma" in model:
          template_file = template_file.replace(
              'config.model_path',
              f'config.model_path = ROOT_DIR + "/models/{model}/"',
          )
        else:
          template_file = template_file.replace(
              'config.model_path',
              'config.model_path ='
              f' ROOT_DIR + "/huggingface/models/{model}"',
          )
        template_file = template_file.replace(
            'config.model_name', f'config.model_name = "{model}"'
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
            'config.prompt_name', f'config.prompt_name = "{prompt_name}"'
        )
        template_file = template_file.replace(
            'config.rule_name', f'config.rule_name = "{RULE}"'
        )
        template_file = template_file.replace(
            'config.num_rule_examples',
            f'config.num_rule_examples = {num_examples}',
        )
        template_file = template_file.replace(
            'config.walk_length', f'config.walk_length = {context_length}'
        )
        template_file = template_file.replace(
            'config.num_questions', 'config.num_questions = 1000'
        )
        template_file = template_file.replace(
            'config.activation_window',
            f'config.activation_window = {activation_window}',
        )
        template_file = template_file.replace(
            'config.layers', ''
        )
        template_file = template_file.replace(
            'config.output_root_folder',
            f'config.output_root_folder = {results_path}',
        )

        # Create new directory for this model, if it doesn't exist
        new_dir = os.path.join(model, prepopulated_path_str)
        os.makedirs(os.path.join(script_dir, new_dir), exist_ok=True)

        # Save off experiment config
        new_filename = f'NTP_d{dimension}_w{world_size}.py'
        new_path = os.path.join(script_dir, new_dir, new_filename)
        new_file = open(new_path, 'w')
        new_file.write(template_file)


if __name__ == '__main__':
  app.run(main)
