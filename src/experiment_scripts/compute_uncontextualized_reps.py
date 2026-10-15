"""Compute uncontextualized representations of each token for a given model.

These uncontextualized representations will be used as a baseline to compare
the relative dirichlet energy of the contextualized representations.
"""

from collections.abc import Sequence
import os
import shutil

from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags
import torch

from ..utils import constants
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

  # Prompt Parameters
  prompt_fn = constants.PROMPTS['next_token_prediction_prepopulated']

  # Make output folder if it doesn't exist
  output_root_folder = config.output_root_folder
  output_folder = os.path.join(
      output_root_folder,
      config.model_name,
  )
  os.makedirs(output_folder, exist_ok=True)

  results = {}

  # Compute uncontextualized representations for each token
  for _, token in enumerate(constants.ITEMS):

    # Generate the NTP prompt, with only single token in the structure
    structure = [token]
    user_prompt, assistant_prompt, _ = prompt_fn(structure)

    # Ensure that the word is encoded as a single token
    assert len(tokenizer.encode(' ' + token, add_special_tokens=False)) == 1

    prompt = model_utils.tokenize(
        model,
        tokenizer,
        [user_prompt],
        prefill_string=assistant_prompt,
    )
    print(f'Token: {tokenizer.decode(prompt.input_ids.tolist()[0][-1])}')

    with torch.inference_mode():
      activations = model(**prompt, output_hidden_states=True).hidden_states
    for layer, layer_activations in enumerate(activations):
      if layer in config.layers:
        if layer not in results.keys():
          results[layer] = {}

        # Get the activations for the final token in the structure
        token_acts = layer_activations[0][-1]

        # Same format as model_utils.get_activations for ease
        results[layer][' ' + token] = {
            'mean_activations': token_acts,
            'num_instances': 1,
        }
      else:
        continue

  torch.save(results, './uncontextualized_reps.pth')

  if os.path.exists(os.path.join(output_folder, 'uncontextualized_reps.pth')):
    os.remove(os.path.join(output_folder, 'uncontextualized_reps.pth'))

  shutil.copy(
      './uncontextualized_reps.pth',
      os.path.join(output_folder, 'uncontextualized_reps.pth'),
  )



if __name__ == '__main__':
  app.run(main)
