"""This script processes the outputs of ICWM open model experiments.

Run this script to generate a csv file containing the metrics for each
experiment.
"""

import os
import shutil

import constants
from absl import app
import numpy as np
import pandas as pd
import torch

from src.utils import analysis_helpers


TASK = "grid"
UNCONTEXTUALIZED_DIR = constants.ROOT_DIR + "/ICWM/results/"
MODELS = ["olmo-2-13b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]
DIMENSIONS = [1, 2]

WORLD_SIZES = [16, 25]
ACTIVATION_WINDOWS = {
    2: 50,
    1: 150,
}
DIM2PROMPT_PREFIXES = {
    1: ["no_information"],
    2: ["no_information"],
}

DIM2WORLD2RULES = {
    2: {
        25: [
            "one_step_vertical_rule",
            "two_step_vertical_rule",
            "three_step_knight_rule",
        ],
        16: ["one_step_vertical_rule", "two_step_vertical_rule"],
    },
    1: {
        25: ["one_step_vertical_rule", "two_step_vertical_rule"],
        16: ["one_step_vertical_rule", "two_step_vertical_rule"],
    },
}

def compute_relative_dirichlet_energy(
    walk_activations, rule_activations, uncontextualized_reps, grid
):
  """Computes the relative Dirichlet energy for a set of activations for a grid.
  
  Args:
    walk_activations: A dictionary mapping tokens to activations for the walk.
    rule_activations: A dictionary mapping tokens to activations for the rule.
    uncontextualized_reps: A dictionary mapping tokens to activations for the
      uncontextualized representations.
    grid: The grid used to generate the data.

  Returns:
    A tuple containing:
      - The Dirichlet energy of the rule activations relative to the walk
        activations.
      - The Dirichlet energy of the uncontextualized representations relative to
        the walk activations.
  """
  # First find the intersection of walk and rule keys
  walk_activations = {
      k: v for k, v in walk_activations.items() if v["num_instances"]
  }
  rule_activations = {
      k: v for k, v in rule_activations.items() if v["num_instances"]
  }
  walk_activations = {
      k: v for k, v in walk_activations.items() if k in rule_activations
  }
  rule_activations = {
      k: v for k, v in rule_activations.items() if k in walk_activations
  }
  uncontextualized_reps = {
      k: v for k, v in uncontextualized_reps.items() if k in walk_activations
  }

  if not walk_activations:
    raise ValueError("No common keys found between walk and rule activations.")

  # Then compute the Dirichlet energy for each common key
  _, walk_energy = analysis_helpers.compute_dirichlet_energy(
      grid, walk_activations
  )
  _, rule_energy = analysis_helpers.compute_dirichlet_energy(
      grid, rule_activations
  )
  _, uncontextualized_energy = analysis_helpers.compute_dirichlet_energy(
      grid, uncontextualized_reps
  )
  return rule_energy / walk_energy, uncontextualized_energy / walk_energy


def main(*_) -> None:

  results_df = {
      "model": [],
      "world_size": [],
      "dimension": [],
      "activation_window": [],
      "prompt": [],
      "rule": [],
      "layer": [],
      "icwm_acc": [],
      "mean_relative_energy": [],
      "std_relative_energy": [],
      "mean_baseline_energy": [],
      "std_baseline_energy": [],
  }

  local_path = "./.cache"
  os.makedirs(local_path, exist_ok=True)

  # Get results from different ICWM experiments
  results_dir = (
      f"{constants.ROOT_DIR}/ICWM/prepopulated_results_structure/"
  )

  for model in MODELS:
    print(model)
    # Load uncontextualized reps for model
    uncontextualized_path = f"{UNCONTEXTUALIZED_DIR}/{model}"
    if os.path.exists(os.path.join(local_path, "uncontextualized_reps.pth")):
      os.remove(os.path.join(local_path, "uncontextualized_reps.pth"))
    shutil.copy(
        os.path.join(uncontextualized_path, "uncontextualized_reps.pth"),
        os.path.join(local_path, "uncontextualized_reps.pth"),
    )
    uncontextualized_reps = torch.load(
        os.path.join(local_path, "uncontextualized_reps.pth"),
        map_location=torch.device("cpu"),
    )
    
    for world_size in WORLD_SIZES:
      for dimension in DIMENSIONS:
        for prompt in DIM2PROMPT_PREFIXES[dimension]:
          for rule in DIM2WORLD2RULES[dimension][world_size]:
            activation_window = ACTIVATION_WINDOWS[dimension]
            print(rule)
            dimension_suffix = "_1d" if dimension == 1 else ""
            storage_path = f"{results_dir}/{model}/world_size_{world_size}/dimension_{dimension}/activation_window_{activation_window}/{prompt}{dimension_suffix}_prepopulated_{rule}"

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
            acc = analysis_helpers.compute_icwm_accuracy(results)

            # Get activations for structure and rule
            if os.path.exists(os.path.join(local_path, "rule_activations.pth")):
              os.remove(os.path.join(local_path, "rule_activations.pth"))

            shutil.copy(
                os.path.join(storage_path, "rule_activations.pth"),
                os.path.join(local_path, "rule_activations.pth"),
            )

            rule_reps = torch.load(
                os.path.join(local_path, "rule_activations.pth"),
                weights_only=False,
            )
            if os.path.exists(
                os.path.join(local_path, "structure_activations.pth")
            ):
              os.remove(
                  os.path.join(local_path, "structure_activations.pth")
              )

            shutil.copy(
                os.path.join(storage_path, "structure_activations.pth"),
                os.path.join(local_path, "structure_activations.pth"),
            )

            walk_reps = torch.load(
                os.path.join(local_path, "structure_activations.pth"),
                weights_only=False,
            )

            grids = results["grid"]

            # Compute relative Dirichlet energy
            layers = set(walk_reps[0].keys()).intersection(
                set(rule_reps[0].keys())
            )
            for layer in layers:
              print(layer)
              layer_energies = []
              layer_baseline_energies = []
              for question_idx, _ in enumerate(walk_reps):
                grid = grids[question_idx]
                walk_activations = walk_reps[question_idx][layer]
                rule_activations = rule_reps[question_idx][layer]
                uncontextualized = uncontextualized_reps[layer]
                relative_energy, baseline_energy = (
                    compute_relative_dirichlet_energy(
                        walk_activations,
                        rule_activations,
                        uncontextualized,
                        grid,
                    )
                )
                layer_energies.append(relative_energy)
                layer_baseline_energies.append(baseline_energy)

              mean_relative_energy = np.mean(layer_energies)
              std_relative_energy = np.std(layer_energies)
              mean_baseline_energy = np.mean(layer_baseline_energies)
              std_baseline_energy = np.std(layer_baseline_energies)

              results_df["model"].append(model)
              results_df["world_size"].append(world_size)
              results_df["dimension"].append(dimension)
              results_df["activation_window"].append(activation_window)
              results_df["prompt"].append(prompt)
              results_df["rule"].append(rule)
              results_df["layer"].append(layer)
              results_df["icwm_acc"].append(acc)
              results_df["mean_relative_energy"].append(mean_relative_energy)
              results_df["std_relative_energy"].append(std_relative_energy)
              results_df["mean_baseline_energy"].append(mean_baseline_energy)
              results_df["std_baseline_energy"].append(std_baseline_energy)


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
