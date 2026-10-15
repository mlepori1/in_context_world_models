"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_name = "gemini-2.5-flash"
  config.gemini_thinking_budget = 0

  config.world_size = 16
  config.dimension = 2
  config.grid_shape = (4, 4)

  config.prompt_name = 'explicit_grid'
  config.rule_name = "two_step_vertical_rule"
  config.num_rule_examples = 6
  config.walk_length = 0
  config.num_questions = 1000

  config.output_root_folder = ROOT_DIR + "/ICWM/results/"

  return config
