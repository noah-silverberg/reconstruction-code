"""
main.py

Main entry point for the MRI reconstruction pipeline.
Reads configuration from config.yaml and executes the pipeline.
"""

import yaml
from pipeline import run_pipeline


def load_config(config_file="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML file.

    Returns
    -------
    dict
        Configuration data.
    """
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    run_pipeline(config)
