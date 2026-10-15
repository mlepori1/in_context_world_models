"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_name
  config.gemini_thinking_budget

  config.world_size
  config.dimension
  config.grid_shape

  config.prompt_name = 'explicit_grid'
  config.rule_name
  config.num_rule_examples
  config.walk_length = 0
  config.num_questions = 1000

  config.output_root_folder

  return config
