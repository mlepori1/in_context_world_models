"""A config describing a world model ICL experiment."""
import ml_collections
from config_constants import ROOT_DIR


def get_config() -> ml_collections.ConfigDict:
  """Get the config for the experiment."""
  config = ml_collections.ConfigDict()

  config.model_path = ROOT_DIR + "/models/gemma-3-4b-it/"
  config.model_name = "gemma-3-4b-it"
  config.output_root_folder = ROOT_DIR + "/ICWM/results/"
  config.layers = [30, 32, 34]

  return config
