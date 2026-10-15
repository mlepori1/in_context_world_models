"""This script processes the outputs of ICWM open model experiments.

Run this script to generate a csv file containing the metrics for each
experiment.
"""

import os
import shutil

import constants
from absl import app
import pandas as pd
import torch

from src.utils import analysis_helpers


TASK = "grid"

MODELS = ["olmo-2-13b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it" ]
DIMENSIONS = [1, 2]

WORLD_SIZES = [16, 25]
ACTIVATION_WINDOWS = {
    2: 50,
    1: 150,
}
DIM2PROMPT_PREFIXES = {
    1: ["no_information", "next_token_prediction", "alternative_next_token_prediction"],
    2: ["no_information", "explicit_grid", "next_token_prediction", "alternative_next_token_prediction"],
}

DIM2WORLD2RULES = {
    2: {
        25: [
            "one_step_vertical_rule",
            "two_step_vertical_rule",
            "three_step_knight_rule",
            "null_rule",
        ],
        16: ["one_step_vertical_rule", "two_step_vertical_rule", "null_rule"],
    },
    1: {
        25: ["one_step_vertical_rule", "two_step_vertical_rule", "null_rule"],
        16: ["one_step_vertical_rule", "two_step_vertical_rule", "null_rule"],
    },
}


def main(*_) -> None:

  results_df = {
      "prepopulated": [],
      "constant_context": [],
      "metalearning": [],
      "model": [],
      "world_size": [],
      "dimension": [],
      "activation_window": [],
      "prompt": [],
      "rule": [],
      "acc": [],
  }

  local_path = "./.cache"
  os.makedirs(local_path, exist_ok=True)

  # Get results from different ICWM experiments
  for constant_context in [True, False]:
    if constant_context:
      cc_suffix = "_constant_context"
    else:
      cc_suffix = ""

    for prepopulated in [True, False]:
      if prepopulated:
        results_prefix = "prepopulated_"
      else:
        results_prefix = ""

      results_dir = (
          f"{constants.ROOT_DIR}/ICWM/{results_prefix}results{cc_suffix}/"
      )

      if not os.path.exists(results_dir):
        print(f"File {results_dir} does not exist.")
        continue

      for model in MODELS:
        for world_size in WORLD_SIZES:
          for dimension in DIMENSIONS:
            for prompt in DIM2PROMPT_PREFIXES[dimension]:
              for rule in DIM2WORLD2RULES[dimension][world_size]:
                for metalearning in [True, False]:
                  if metalearning:
                    metalearning_prefix = "metalearning_"
                  else:
                    metalearning_prefix = ""

                  activation_window = ACTIVATION_WINDOWS[dimension]
                  print(rule)
                  dimension_suffix = "_1d" if dimension == 1 else ""
                  if prepopulated:
                    prepopulated_suffix = "_prepopulated"
                  else:
                    prepopulated_suffix = ""
                  storage_path = f"{results_dir}/{model}/world_size_{world_size}/dimension_{dimension}/activation_window_{activation_window}/{metalearning_prefix}{prompt}{dimension_suffix}{prepopulated_suffix}_{rule}"

                  if not os.path.exists(os.path.join(storage_path, "results.pth")):
                    print(
                        f"File {os.path.join(storage_path, 'results.pth')} "
                        "does not exist."
                    )
                    continue

                  if os.path.exists(os.path.join(local_path, "results.pth")):
                    os.remove(os.path.join(local_path, "results.pth"))

                  shutil.copy(
                      os.path.join(storage_path, "results.pth"),
                      os.path.join(local_path, "results.pth"),
                  )

                  results = torch.load(
                      os.path.join(local_path, "results.pth"), weights_only=False
                  )

                  # Compute metrics
                  if "next_token_prediction" in prompt:
                    acc = analysis_helpers.compute_icwm_accuracy(
                        results, ntp=True
                    )
                  else:
                    acc = analysis_helpers.compute_icwm_accuracy(results)

                  results_df["prepopulated"].append(prepopulated)
                  results_df["constant_context"].append(constant_context)
                  results_df["metalearning"].append(metalearning)
                  results_df["model"].append(model)
                  results_df["world_size"].append(world_size)
                  results_df["dimension"].append(dimension)
                  results_df["activation_window"].append(activation_window)
                  results_df["prompt"].append(prompt)
                  results_df["rule"].append(rule)
                  results_df["acc"].append(acc)

      # Save collated results to CSV
      if os.path.exists(os.path.join(local_path, "collated_results.csv")):
        os.remove(os.path.join(local_path, "collated_results.csv"))
      pd.DataFrame().from_dict(results_df).to_csv(
          os.path.join(local_path, "collated_results.csv"), index=False
      )

      # Copy collated results to storage
      if os.path.exists(f"{results_dir}/collated_exp1_{TASK}_results.csv"):
        os.remove(f"{results_dir}/collated_exp1_{TASK}_results.csv")
      shutil.copy(
          os.path.join(local_path, "collated_results.csv"),
          os.path.join(f"{results_dir}/collated_exp1_{TASK}_results.csv"),
      )


if __name__ == "__main__":
  app.run(main)
