"""
Code to run a swiss tournament between OpenReview papers using LLM as the underlying comparison model
"""

import argparse
import json
import os

import torch
import yaml

from anthropic import Anthropic
from openai import OpenAI

from automatic_scientific_qm.utils.data import Paper
from automatic_scientific_qm.llm_reviewing.swiss_tournament import tournament_ranking

"""
This code includes portions from https://github.com/NoviScl/AI-Researcher by Chenglei Si,
licensed under the MIT License (see below or at https://github.com/NoviScl/AI-Researcher?tab=MIT-1-ov-file#readme).
Copyright (c) 2024 Chenglei Si
"""


def run_llm_ranking(config: dict) -> None:
    """
    Runs an X round swiss tournamnet as a ranking process based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        None
    """

    # Load papers
    if config["dataset"] == "iclr":
        samples = torch.load("../../data/iclr_200_subset_data.pt")
    elif config["dataset"] == "neurips":
        samples = torch.load("../../data/neurips_200_subset_data.pt")
    else:
        raise NotImplementedError("Only iclr and neurips are supported")

    paper_directory = "../../data/openreview-dataset-private/labelled_papers"
    paperhash2sample = {}
    papers = []
    for sample in samples:
        paperhash = sample["paperhash"]
        paperhash2sample[paperhash] = sample
        with open(os.path.join(paper_directory, f"{paperhash}.json")) as file:
            paper = json.load(file)
            paper = Paper(**paper)
            papers.append(paper)

    tournament_ranking(
        papers,
        config["model"],
        config["seed"],
        config["dataset"],
        paperhash2sample,
        max_round=5,
        format="json",
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    run_llm_ranking(config)
