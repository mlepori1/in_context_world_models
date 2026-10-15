"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_name = "claude-3-5-sonnet-latest"
  config.gemini_thinking_budget = 5000

  config.world_size = 16
  config.dimension = 1
  config.grid_shape = (16,)

  config.prompt_name = "next_token_prediction_1d"
  config.rule_name = "null_rule"
  config.num_rule_examples = 0
  config.walk_length = 1500
  config.num_questions = 1000

  config.output_root_folder = ROOT_DIR + "/ICWM/results/"

  return config
