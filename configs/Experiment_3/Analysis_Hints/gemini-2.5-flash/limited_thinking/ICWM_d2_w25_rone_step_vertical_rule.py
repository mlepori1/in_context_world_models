"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_name = "gemini-2.5-flash"
  config.gemini_thinking_budget = 0

  config.world_size = 25
  config.dimension = 2
  config.grid_shape = (5, 5)

  config.prompt_name = "high_information"
  config.rule_name = "one_step_vertical_rule"
  config.num_rule_examples = 10
  config.walk_length = 1500
  config.num_questions = 1000

  config.output_root_folder = ROOT_DIR + "/ICWM/results/"

  return config
