"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_path
  config.model_name
  config.world_size
  config.dimension
  config.grid_shape
  config.prompt_name
  config.rule_name
  config.num_rule_examples
  config.walk_length
  config.num_questions
  config.activation_window
  config.output_root_folder

  return config
