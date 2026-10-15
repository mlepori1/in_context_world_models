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

DIMENSION = [2]
WORLD_SIZES = [16, 25]

Dimension2WorldSize2Rules = {
    2: {
        16: ['one_step_vertical_rule', 'two_step_vertical_rule'],
        25: [
            'one_step_vertical_rule',
            'two_step_vertical_rule',
            'three_step_knight_rule',
        ],
    },
}

Dimension2ActivationWindows = {
    2: 50,
}

WorldSize2Rules2NumExamples = {
    16: {
        'one_step_vertical_rule': 10,
        'two_step_vertical_rule': 6,
    },
    25: {
        'one_step_vertical_rule': 10,
        'two_step_vertical_rule': 10,
        'three_step_knight_rule': 10,
    },
}


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  script_dir = os.path.dirname(os.path.abspath(__file__))
  print(script_dir)

  for model in MODELS:
    for dimension in DIMENSION:
      for world_size in WORLD_SIZES:
        for rule in Dimension2WorldSize2Rules[dimension][world_size]:
          # Get specific parameters for this experiment
          num_examples = WorldSize2Rules2NumExamples[world_size][rule]
          activation_window = Dimension2ActivationWindows[dimension]

          grid_shape = (
              int(math.sqrt(world_size)),
              int(math.sqrt(world_size)),
          )

          context_length = 0
          prompt_name = 'explicit_grid_prepopulated'
          results_path = "ROOT_DIR + '/ICWM/prepopulated_results/'"

          # Load template
          template_path = os.path.join(script_dir, 'template.py')
          template_file = open(template_path, 'r').read()

          # Edit template
          if 'gemma' in model:
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
              'config.rule_name', f'config.rule_name = "{rule}"'
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
          new_dir = os.path.join(script_dir, model)
          os.makedirs(new_dir, exist_ok=True)

          # Save off experiment config
          new_filename = f'ICWM_Grid_d{dimension}_w{world_size}_{rule}.py'
          new_path = os.path.join(new_dir, new_filename)
          new_file = open(new_path, 'w')
          new_file.write(template_file)


if __name__ == '__main__':
  app.run(main)
