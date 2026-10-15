"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_path = ROOT_DIR + "/models/gemma-3-27b-it/"
  config.model_name = "gemma-3-27b-it"
  config.world_size = 16
  config.dimension = 1
  config.grid_shape = (16,)
  config.prompt_name = "next_token_prediction_1d"
  config.rule_name = "null_rule"
  config.num_rule_examples = 0
  config.walk_length = 1800
  config.num_questions = 1000
  config.activation_window = 150
  config.output_root_folder = ROOT_DIR + "/ICWM/results/"

  return config
