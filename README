# Licensing & Disclaimer
Copyright 2026 Google LLC  

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.


# In-Context World Models

## Guide to Replicating Results using an Alternative Next Token Prediction Prompt

### Step 1: Run ICLR Analysis and find best context length

In `configs/Experiment_1/Alternative_ICLR`, there is a set of config files that define an ICLR analysis using the new next token prediction prompt. First, run these configs by running bash scripts like the following for each model: 

```
for config_file in configs/Experiment_1/Alternative_ICLR/gemma_27b/*.py; do \
  echo "Processing $config_file"; \
  python3 src/experiment_scripts/run_iclr_experiments.py --config=$config_file; done
```

This will kick off an analysis for each topology, for each model. The results will be saved in `{ROOT_DIR}/ICWM/results/gemma-3-27b-it/world_size_25/dimension_2/activation_window_50/alternative_next_token_prediction/representation_metrics.csv`.

In general, the ICLR analysis results are stored in `{ROOT_DIR}/ICWM/results/{model}/{world_size}/{dimension}/{activation_window}/{prompt_type}/representation_metrics.csv`. Because we are using a different prompt, we use `alternative_next_token_prediction` as the `prompt_type`, and this should not overwrite any prior results. Activation window is 50 for 2D grids, 150 for 1d lines.


Next, we must actually figure out the best context length to use. To do this, we can use the colab found in `analysis/ICLR_Analysis.ipynb`. This colab has functions for finding the best context length using a either normalized dirichlet energy or distance correlation. In general, `distance_correlation` is found to perform better, as it is a bit more interpretable and less subject to noise in the beginning of the context (especially for models that overall perform poorly).

There is a @todo section in the colab that you'll have to run to get the best context length for each model for the alternative next token prediction prompt.


### Step 2: Run Next Token Prediction Evaluation Using Best Context Length

Now that we know the best context length, we can run a next token prediction evaluation. To do this, we have to generate a set of config files `configs/Experiment_1/Alternative_NTP`. Currently, you will find a 
`generate_configs.py` file that can be used to generate the config files for you. **Before you run this script, you'll have to fill out the fields in `Model2Dimension2WorldSize2ContextLength`, corresponding to the best context length found in the previous step for each model**. 
Once you do this, you can use blaze to run the script.

Finally, you'll have to kick off an xmanager job using the generated config files. You can do this using a command like the following:

```
shopt -s globstar

BASE_PATH="configs/Experiment_1/Alternative_NTP/"

echo "Starting search for configs in $BASE_PATH..."

for config_file in "$BASE_PATH"/*/*/*.py; do
  if [ ! -f "$config_file" ]; then
    echo "No .py files found matching the pattern."
    continue
  fi

  echo "---------------------------------"
  echo "Processing $config_file"
  echo "---------------------------------"
  python3 src/experiment_scripts/run_open_model_experiments.py --config="$config_file"
done

echo "All experiments launched."
```

Results will be stored in `{ROOT_DIR}/ICWM/results/{model}/{world_size}/{dimension}/{activation_window}/alternative_next_token_prediction_null_rule/`.

For all non-ICLR results, we append a string to define which type of rule we use to generate few shot examples. In this case, we use the null rule, which means we generate no few shot examples.


### Step 3: Run ICWM Evaluation Using Best Context Length
Next, we want to run an evaluation of the ICWM using the best context length found in step 1. To do this, we'll use a similar process as before. First, we'll create a set of config files that define the ICWM evaluation. Currently, there is a file `generate_configs.py` that creates these config files. This time, you'll have to fill out the fields in `AlternativeModel2Dimension2WorldSize2ContextLength`, corresponding to the best context length found in step 1 for each model. 

This will generate a set of configs in the `alternative_not_prepopulated` subdirectory of `configs/Experiment_2/ICWM/{model}`. Run these configs using a command like the following:
```
shopt -s globstar

BASE_PATH="configs/Experiment_2/ICWM/gemma-3-27b-it/alternative_not_prepopulated/"

echo "Starting search for configs in $BASE_PATH..."

for config_file in "$BASE_PATH"/*.py; do
  if [ ! -f "$config_file" ]; then
    echo "No .py files found matching the pattern."
    continue
  fi

  echo "---------------------------------"
  echo "Processing $config_file"
  echo "---------------------------------"
  python3 src/experiment_scripts/run_open_model_experiments.py --config="$config_file"
done

echo "All experiments launched."
```

**Note: Running these jobs will overwrite the results from running the `not_prepopulated` jobs in the same directory. This is fine, as the alternative prompt will be the the main analyais prompt in the paper. Just make sure that the jobs actually complete (or delete the original results), so we don't mix results from incomplete jobs. However, I wouldn't necessarily make a new directory, unless you want to change the file that collates results: `analysis/process_open_models.py`. Up to you!** 

**Note 2: In the paper, we mainly analyze prepopulated results for ICWM, as everything performs better with prepopulation. However, for thoroughness, we also want to try not prepopulating. We do this JUST FOR GEMMA 27B, as it will be an appendix point anyway. Do not worry if you don't see alternative not prepopulated configs for other models.**

Results will be stored in `{ROOT_DIR}/ICWM/results/{model}/world_size_{world_size}/dimension_{dimension}/activation_window_{activation_window}/no_information_{rule}/`.


### Step 4: Run API Model Evaluation

Next, we want to run evaluations of the API model. To do this, simply run the configs I've already generated in `configs/Experiment_3`.

Here is an example command.

```
shopt -s globstar

BASE_PATH="configs/Experiment_3/Alternative_API"

echo "Starting search for configs in $BASE_PATH..."

for config_file in "$BASE_PATH"/gemini-2.5-pro/limited_thinking/ICWM_d2_w25_rtwo_step_vertical_rule.py; do
  if [ ! -f "$config_file" ]; then
    echo "No .py files found matching the pattern."
    continue
  fi

  echo "---------------------------------"
  echo "Processing $config_file"
  echo "---------------------------------"
  
  python3 src/experiment_scripts/run_api_experiments.py --config="$config_file"
done

echo "All experiments launched."
```

### Step 5: Run Appendix Results

Finally,there's some extra experiments hiding in the appendix that we should run! These are just next token prediction experiments with a constant, long context (1500 tokens). Find them in `configs/Appendix/Alternative_NTP_Constant_Context`. You can run these with a command like the following:

```
shopt -s globstar

BASE_PATH="configs/Appendix/Alternative_NTP_Constant_Context/"

echo "Starting search for configs in $BASE_PATH..."

for config_file in "$BASE_PATH"/*/*/*.py; do
  if [ ! -f "$config_file" ]; then
    echo "No .py files found matching the pattern."
    continue
  fi

  echo "---------------------------------"
  echo "Processing $config_file"
  echo "---------------------------------"
  python3 src/experiment_scripts/run_open_model_experiments.py --config="$config_file"
done

echo "All experiments launched."
```

Results will be stored in `{ROOT_DIR}/ICWM/results_constant_context/{model}/{world_size}/{dimension}/{activation_window}/alternative_next_token_prediction_null_rule/`.

