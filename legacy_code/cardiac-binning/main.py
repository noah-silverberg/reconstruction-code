"""
main.py

Main entry point for the MRI reconstruction pipeline.
Reads configuration from config.yaml and executes the pipeline.
Usage:
    python3 main.py
"""

import yaml
from pipeline import run_pipeline


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    config = load_config()
    run_pipeline(config)
