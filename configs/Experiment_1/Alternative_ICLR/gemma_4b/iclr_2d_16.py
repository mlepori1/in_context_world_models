"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_path = ROOT_DIR + "/models/gemma-3-4b-it/"
  config.model_name = "gemma-3-4b-it"
  config.world_size = 16
  config.dimension = 2
  config.grid_shape = (4, 4)
  config.prompt_name = "alternative_next_token_prediction"
  config.walk_length = 2000
  config.num_questions = 50
  config.activation_window = 50
  config.layers = [1, 30, 32, 34]
  config.output_root_folder = ROOT_DIR + "/ICWM/results/"
  config.get_activations = True

  return config
