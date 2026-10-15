"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_path = ROOT_DIR + "/huggingface/models/olmo-2-13b-it"
  config.model_name = "olmo-2-13b-it"
  config.world_size = 25
  config.dimension = 2
  config.grid_shape = (5, 5)
  config.prompt_name = "next_token_prediction_prepopulated"
  config.walk_length = 2000
  config.num_questions = 50
  config.activation_window = 50
  config.layers = [1, 36, 38, 40]
  config.output_root_folder = ROOT_DIR + "/ICWM/results/"
  config.get_activations = True


  return config
