"""
Data loading and processing utilities.
"""

import os

import numpy as np
import torch

from adapters import AutoAdapterModel
from datasets import load_dataset, load_from_disk
from datasets import Dataset as HF_Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

EPSILON = 1e-3


class ScorePredictionDataset(Dataset):
    def __init__(
        self, dataset: HF_Dataset, score_type: str, paper_representation: str
    ) -> None:
        self.dataset = dataset
        self.score_type = score_type
        self.paper_representation = paper_representation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.dataset[idx]

        paper_representation = sample[self.paper_representation]
        full_paper = self.get_full_paper(sample)

        paper_representation = torch.tensor(paper_representation)
        references = torch.tensor(sample["reference_emb"])
        full_paper = torch.tensor(full_paper)

        return (
            sample[self.score_type],
            paper_representation,
            references,
            full_paper,
        )

    def get_full_paper(self, sample: dict) -> np.ndarray:
        full_paper = np.stack(
            [
                sample["intro_emb"],
                sample["method_emb"],
                sample["result_experiment_emb"],
                sample["background_emb"],
                sample["conclusion_emb"],
            ],
            axis=0,
        )

        return full_paper


def collate(batch: list) -> dict:
    scores = []
    paper_representations = []
    references = []
    full_papers = []
    n_references = []
    for score, paper_representation, reference, full_paper in batch:
        scores.append(score)
        paper_representations.append(paper_representation)
        references.append(reference)
        n_references.append(reference.shape[0])
        full_papers.append(full_paper)

    scores = torch.tensor(scores, dtype=torch.float)

    return {
        "scores": scores,
        "paper_representations": paper_representations,
        "references": references,
        "full_papers": full_papers,
        "n_references": n_references,
    }


def get_data(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary containing data parameters.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: A tuple of three DataLoaders for training, validation, and testing.
    """

    # Check whether embeddings need to be computed
    dataset_directory = os.path.join(
        config["data"]["output_directory"], config["data"]["dataset"]
    )
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    compute_embeddings = not os.path.exists(
        os.path.join(dataset_directory, "dataset_dict.json")
    )

    if compute_embeddings:
        dataset = load_dataset(
            "nhop/scientific-quality-score-prediction", config["data"]["dataset"]
        )

        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base").to(
            config["device"]
        )
        model.load_adapter(
            "allenai/specter2", source="hf", load_as="classification", set_active=True
        )
        model = model.to(config["device"])

        def compute_embeddings(sample):
            # Embed paper sections
            mask = []
            texts = []
            keys = [
                "introduction",
                "background",
                "methodology",
                "experiments_results",
                "conclusion",
            ]

            for key in keys:
                if sample[key] is None:
                    texts.append("None")
                    mask.append(0)
                else:
                    texts.append(sample[key])
                    mask.append(1)

            if sample["title"] is None or sample["abstract"] is None:
                texts.append("None")
                mask.append(0)
            else:
                title_abstract = sample["title"] + " " + sample["abstract"]
                texts.append(title_abstract)
                mask.append(1)

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(config["device"])

            with torch.no_grad():
                outputs = model(**inputs)
            outputs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            embeddings = np.zeros((6, 768))
            mask = np.array(mask, dtype=bool)
            embeddings[mask] = outputs[mask]

            intro_emb = embeddings[0]
            background_emb = embeddings[1]
            method_emb = embeddings[2]
            result_experiment_emb = embeddings[3]
            conclusion_emb = embeddings[4]
            title_abstract_emb = embeddings[5]

            if not mask[-1]:
                title_abstract_emb = None

            # Embed references

            if sample["references"] is None or len(sample["references"]["title"]) == 0:
                reference_emb = np.zeros((1, 768), dtype=np.float32)
            else:
                reference_texts = []
                for title, abstract in zip(
                    sample["references"]["title"], sample["references"]["abstract"]
                ):
                    reference_text = title + abstract
                    reference_texts.append(reference_text)

                inputs = tokenizer(
                    reference_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(config["device"])

                with torch.no_grad():
                    outputs = model(**inputs)

                reference_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Take log of citations per month
            if sample["avg_citations_per_month"] is not None:
                avg_citations_per_month = np.log(
                    sample["avg_citations_per_month"] + EPSILON
                )
            else:
                avg_citations_per_month = None

            return {
                "reference_emb": reference_emb,
                "title_abstract_emb": title_abstract_emb,
                "intro_emb": intro_emb,
                "method_emb": method_emb,
                "result_experiment_emb": result_experiment_emb,
                "background_emb": background_emb,
                "conclusion_emb": conclusion_emb,
                "avg_citations_per_month": avg_citations_per_month,
            }

        dataset = dataset.map(compute_embeddings, batched=False)
        dataset.save_to_disk(dataset_directory)

    else:
        dataset = load_from_disk(dataset_directory)

    # Apply filters
    paper_representation = f"{config['data']['paper_representation']}_emb"
    dataset = dataset.filter(lambda x: x[config["data"]["score_type"]] is not None)
    dataset = dataset.filter(
        lambda x: x[paper_representation] is not None
        and not all([elem == 0 for elem in x[paper_representation]])
    )

    if config["data"]["context_type"] == "references":
        dataset = dataset.filter(
            lambda x: not all([elem == 0 for elem in x["reference_emb"]])
        )

    # Create dataloaders
    train_dataset = ScorePredictionDataset(
        dataset["train"],
        config["data"]["score_type"],
        paper_representation,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate,
    )

    val_dataset = ScorePredictionDataset(
        dataset["validation"],
        config["data"]["score_type"],
        paper_representation,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate,
    )

    test_dataset = ScorePredictionDataset(
        dataset["test"],
        config["data"]["score_type"],
        paper_representation,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate,
    )

    return train_dataloader, val_dataloader, test_dataloader
