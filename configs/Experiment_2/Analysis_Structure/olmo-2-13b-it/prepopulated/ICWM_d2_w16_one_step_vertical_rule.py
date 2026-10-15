"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_path = ROOT_DIR + "/huggingface/models/olmo-2-13b-it"
  config.model_name = "olmo-2-13b-it"
  config.world_size = 16
  config.dimension = 2
  config.grid_shape = (4, 4)
  config.prompt_name = "no_information_prepopulated"
  config.rule_name = "one_step_vertical_rule"
  config.num_rule_examples = 10
  config.walk_length = 100
  config.num_questions = 50
  config.activation_window = 50
  config.layers = [38, 40]
  config.output_root_folder = ROOT_DIR + "/ICWM/prepopulated_results_structure/"
  config.get_activations = True

  return config
