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
  config.prompt_name = "no_information_1d_prepopulated"
  config.rule_name = "one_step_vertical_rule"
  config.num_rule_examples = 10
  config.walk_length = 1500
  config.num_questions = 50
  config.activation_window = 150
  config.layers = [32, 34]
  config.output_root_folder = ROOT_DIR + "/ICWM/prepopulated_results_structure/"
  config.get_activations = True

  return config
