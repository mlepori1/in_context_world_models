"""This script processes the outputs of ICWM API model experiments.

Run this script to generate a csv file containing the metrics for each
experiment.
"""

import math
import os
import pickle as pkl
import shutil

import constants
from absl import app
import pandas as pd

from src.utils import analysis_helpers


TASK = "grid"

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "claude-haiku-4-5-20251001",
]
DIMENSIONS = [1, 2]

WORLD_SIZES = [16, 25]

DIM2PROMPT_PREFIXES = {
    1: ["no_information", "next_token_prediction"],
    2: [
        "no_information",
        "explicit_grid",
        "next_token_prediction",
        "high_information",
        "construct_structure",
    ],
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


MODEL2THINKING_BUDGET = {
    "gemini-2.5-flash": [0, 5000],
    "gemini-2.5-pro": [128, 5000],
    "openai/gpt-5": [-1],
    "openai/gpt-5-mini": [-1],
    "claude-haiku-4-5-20251001": [-1],
}

MODEL2DIR = {
    "gemini-2.5-flash": "results",
    "gemini-2.5-pro": "results",
    "openai/gpt-5": "3rd_party_results",
    "openai/gpt-5-mini": "3rd_party_results",
    "claude-haiku-4-5-20251001": "3rd_party_results",
}

outdir = constants.ROOT_DIR + "/ICWM/results/"


def main(*_) -> None:

  results_df = {
      "thinking": [],
      "model": [],
      "world_size": [],
      "dimension": [],
      "prompt": [],
      "rule": [],
      "acc": [],
  }

  local_path = "./.cache"
  os.makedirs(local_path, exist_ok=True)

  # Get results from different ICWM experiments
  for model in MODELS:
    for thinking_budget in MODEL2THINKING_BUDGET[model]:

      results_dir = (
          f"{constants.ROOT_DIR}/ICWM/{MODEL2DIR[model]}/"
      )

      if not os.path.exists(results_dir):
        print(f"File {results_dir} does not exist.")
        continue

      for world_size in WORLD_SIZES:
        for dimension in DIMENSIONS:
          for prompt in DIM2PROMPT_PREFIXES[dimension]:
            for rule in DIM2WORLD2RULES[dimension][world_size]:
              print(rule)
              dimension_suffix = "_1d" if dimension == 1 else ""

              storage_path = f"{results_dir}/{model}/thinking_{thinking_budget}/world_size_{world_size}/dimension_{dimension}/{prompt}{dimension_suffix}_{rule}"

              if not os.path.exists(os.path.join(storage_path, "results.pkl")):
                print(
                    f"File {os.path.join(storage_path, 'results.pkl')} "
                    "does not exist."
                )
                continue

              if os.path.exists(os.path.join(local_path, "results.pkl")):
                os.remove(os.path.join(local_path, "results.pkl"))

              shutil.copy(
                  os.path.join(storage_path, "results.pkl"),
                  os.path.join(local_path, "results.pkl"),
              )

              results = pkl.load(
                  open(os.path.join(local_path, "results.pkl"), "rb"),
              )

              # Compute metrics
              if "next_token_prediction" in prompt:
                acc = analysis_helpers.compute_icwm_accuracy(results, ntp=True)
              elif "construct_structure" in prompt:
                grid_shape = (
                    int(math.sqrt(world_size)),
                    int(math.sqrt(world_size)),
                )
                acc = analysis_helpers.compute_structure_accuracy(
                    results, grid_shape
                )
              else:
                acc = analysis_helpers.compute_icwm_accuracy(results)

              results_df["thinking"].append(thinking_budget)
              results_df["model"].append(model)
              results_df["world_size"].append(world_size)
              results_df["dimension"].append(dimension)
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
      if os.path.exists(f"{outdir}/collated_exp3_{TASK}_results.csv"):
        os.remove(f"{outdir}/collated_exp3_{TASK}_results.csv")
      shutil.copy(
          os.path.join(local_path, "collated_results.csv"),
          os.path.join(f"{outdir}/collated_exp3_{TASK}_results.csv"),
      )


if __name__ == "__main__":
  app.run(main)
