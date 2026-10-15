"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_path = ROOT_DIR + "/models/gemma-3-4b-it/"
  config.model_name = "gemma-3-4b-it"
  config.world_size = 25
  config.dimension = 1
  config.grid_shape = (25,)
  config.prompt_name = "alternative_next_token_prediction_1d"
  config.rule_name = "null_rule"
  config.num_rule_examples = 0
  config.walk_length = 1500
  config.num_questions = 1000
  config.activation_window = 150
  config.output_root_folder = ROOT_DIR + "/ICWM/results_constant_context/"

  return config
