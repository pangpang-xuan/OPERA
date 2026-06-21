# OPERA


## 📌 Overview

The repository is structured as follows to support both cold-start data generation and Reinforcement Learning (RL) training loops:

* **`sft_data_sync/reer_tot_parallel_ppl.py`**: The core pipeline for generating **Cold-Start SFT data** utilizing parallel processing strategies.
* **`verl/sh_rethinking_grpo.sh`**: The main entry script for **Reinforcement Learning (RL) training**, powered by the VeRL framework and optimized via Group Relative Policy Optimization (GRPO).
* **`verl/all2hf.sh`**: A post-processing utility script to convert the trained RL model checkpoints into standard **Hugging Face (HF)** format for evaluation and deployment.

> 📝 **Note on Benchmarks:** Standard evaluation and benchmark code are excluded from this repository to maintain anonymity/repository hygiene. Please download and run them directly from the official respective repositories.

> Data Availability: The complete, full-scale dataset will be fully open-sourced upon the acceptance of the paper.

