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

RULES = [
    'one_step_vertical_rule',
    'two_step_vertical_rule',
    'three_step_knight_rule',
]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  script_dir = os.path.dirname(os.path.abspath(__file__))

  for model in MODELS:
    for rule in RULES:
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
          'config.rule_name', f'config.rule_name = "{rule}"'
      )

      # Create new directory for this model, if it doesn't exist
      new_dir = model
      os.makedirs(os.path.join(script_dir, new_dir), exist_ok=True)

      # Save off experiment config
      new_filename = f'ICWM_{rule}.py'
      new_path = os.path.join(script_dir, new_dir, new_filename)
      new_file = open(new_path, 'w')
      new_file.write(template_file)


if __name__ == '__main__':
  app.run(main)
