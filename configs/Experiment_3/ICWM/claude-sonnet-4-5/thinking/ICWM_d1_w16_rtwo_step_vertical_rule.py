"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_name = "claude-sonnet-4-5-20250929"
  config.gemini_thinking_budget = -1

  config.world_size = 16
  config.dimension = 1
  config.grid_shape = (16,)

  config.prompt_name = "no_information_1d"
  config.rule_name = "two_step_vertical_rule"
  config.num_rule_examples = 6
  config.walk_length = 1500
  config.num_questions = 1000

  config.output_root_folder = ROOT_DIR + "/ICWM/3rd_party_results/"

  return config
