"""
PyTorch Lightning Trainer for the score prediction models.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

from automatic_scientific_qm.score_prediction.model import ScoreModel


class ScorePrediction(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()

        self.context_type = config["data"]["context_type"]
        if config["data"]["context_type"] == "no_context":
            use_context = False
        else:
            use_context = True
        self.score_model = ScoreModel(config["trainer"]["model"], use_context)
        self.loss = nn.L1Loss()
        self.lr = config["trainer"]["lr"]
        self.optimizer = self.configure_optimizers()
        self.save_hyperparameters()

    def get_loss(self, data: dict) -> torch.Tensor:
        papers, masks, scores = self.transform_data(data)
        predicted_scores = self.score_model(papers, masks).squeeze(1)
        loss = self.loss(predicted_scores, scores)
        return loss

    def training_step(self, data: dict, data_idx: int) -> torch.Tensor:
        loss = self.get_loss(data)
        self.log("training/loss", loss)
        return loss

    def validation_step(self, data: dict, data_idx: int) -> None:
        loss = self.get_loss(data)
        self.log("validation/loss", loss)

    def predict(self, data: dict) -> torch.Tensor:
        papers, masks, _ = self.transform_data(data)
        predicted_scores = self.score_model(papers, masks).squeeze(1)
        return predicted_scores

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def transform_data(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms the input data into tensors for model training.

        Args:
            data (dict): The input data dictionary containing the following keys:
                - "paper_representations" (list): List of paper representations.
                - "references" (list): List of reference embeddings.
                - "full_papers" (list): List of full paper embeddings.
                - "scores" (torch.Tensor): Tensor of target scores.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the following tensors:
                - paper (torch.Tensor): Tensor of input representations.
                - masks (torch.Tensor): Tensor of masks indicating the valid positions in the input representations.
                - scores (torch.Tensor): Tensor of target scores.
        """
        input_lengths = []

        if self.context_type == "no_context":
            input_representations = torch.stack(data["paper_representations"], dim=0)
            masks = torch.zeros(
                input_representations.shape[0], input_representations.shape[1]
            )

        elif self.context_type == "references":
            for paper_emb, context_emb in zip(
                data["paper_representations"], data["references"]
            ):
                input_representation = torch.cat(
                    [paper_emb.unsqueeze(0), context_emb], dim=0
                )
                input_lengths.append(input_representation.shape[0])

            input_representations = pad_sequence(
                input_representations, batch_first=True
            )
            B, T, _ = input_representations.shape
            input_lengths = torch.tensor(input_lengths)
            masks = ~(torch.arange(T).expand(B, T) < input_lengths.unsqueeze(1))

        elif self.context_type == "full_paper":
            for paper_emb, context_emb in zip(
                data["paper_representations"], data["full_papers"]
            ):
                input_representation = torch.cat(
                    [paper_emb.unsqueeze(0), context_emb], dim=0
                )
                input_representations.append(input_representation)

            input_representations = torch.stack(input_representations, dim=0)
            masks = torch.zeros(
                input_representations.shape[0], input_representations.shape[1]
            )
        else:
            raise NotImplementedError(
                f"The context type {self.context_type} is not implemented."
            )

        paper = input_representations.to(data["scores"])
        masks = masks.to(data["scores"])

        return paper, masks, data["scores"]


class PairwiseComparison(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.context_type = config["data"]["context_type"]
        if config["data"]["context_type"] == "no_context":
            use_context = False
        else:
            use_context = True
        self.score_model = ScoreModel(config["trainer"]["model"], use_context)
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = config["trainer"]["lr"]
        self.optimizer = self.configure_optimizers()
        self.save_hyperparameters()

    def get_loss(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        papers1, masks1, papers2, masks2, scores = self.transform_data(data)
        predicted_scores1 = self.score_model(papers1, masks1).squeeze(1)
        predicted_scores2 = self.score_model(papers2, masks2).squeeze(1)
        score_diff = predicted_scores1 - predicted_scores2
        loss = self.loss(score_diff, scores)
        acc = ((score_diff > 0) == scores).float().mean()
        return loss, acc

    def training_step(self, data: dict, data_idx: int) -> torch.Tensor:
        loss, acc = self.get_loss(data)
        self.log("training/loss", loss)
        self.log("training/acc", acc)
        return loss

    def validation_step(self, data: dict, data_idx: int) -> torch.Tensor:
        loss, acc = self.get_loss(data)
        self.log("validation/loss", loss)
        self.log("validation/acc", acc)
        return loss

    def predict(self, data: dict) -> torch.Tensor:
        papers1, masks1, papers2, masks2, _ = self.transform_data(data)
        predicted_scores1 = self.score_model(papers1, masks1).squeeze(1)
        predicted_scores2 = self.score_model(papers2, masks2).squeeze(1)
        return predicted_scores1 >= predicted_scores2

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def transform_data(
        self, data: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms the input data into tensors for model training.

        Args:
            data (dict): The input data dictionary containing the following keys:
                - "paper_representations" (list): List of torch.Tensor representing the paper representations.
                - "references" (list): List of torch.Tensor representing the reference embeddings.
                - "full_papers" (list): List of torch.Tensor representing the full paper embeddings.
                - "scores" (torch.Tensor): Tensor of scores.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the following tensors:
                - papers1 (torch.Tensor): Tensor of input representations for the first set of papers.
                - masks1 (torch.Tensor): Tensor of masks for the first set of papers.
                - papers2 (torch.Tensor): Tensor of input representations for the second set of papers.
                - masks2 (torch.Tensor): Tensor of masks for the second set of papers.
                - scores (torch.Tensor): Tensor of scores indicating the preference between the two sets of papers.
        """
        input_representations = []
        input_lengths = []

        if self.context_type == "no_context":
            input_representations = torch.stack(data["paper_representations"], dim=0)
            masks = torch.zeros(
                input_representations.shape[0], input_representations.shape[1]
            )

        elif self.context_type == "references":
            for paper_emb, context_emb in zip(
                data["paper_representations"], data["references"]
            ):
                input_representation = torch.cat(
                    [paper_emb.unsqueeze(0), context_emb], dim=0
                )
                input_representations.append(input_representation)
                input_lengths.append(input_representation.shape[0])

            input_representations = pad_sequence(
                input_representations, batch_first=True
            )
            B, T, _ = input_representations.shape
            input_lengths = torch.tensor(input_lengths)
            masks = ~(torch.arange(T).expand(B, T) < input_lengths.unsqueeze(1))

        elif self.context_type == "full_paper":
            for paper_emb, context_emb in zip(
                data["paper_representations"], data["full_papers"]
            ):
                input_representation = torch.cat(
                    [paper_emb.unsqueeze(0), context_emb], dim=0
                )
                input_representations.append(input_representation)

            input_representations = torch.stack(input_representations, dim=0)
            masks = torch.zeros(
                input_representations.shape[0], input_representations.shape[1]
            )
        else:
            raise NotImplementedError(
                f"The context type {self.context_type} is not implemented."
            )

        n_samples = len(data["scores"]) // 2
        papers1 = input_representations[:n_samples].to(data["scores"])
        papers2 = input_representations[n_samples : 2 * n_samples].to(data["scores"])
        masks1 = masks[:n_samples].to(data["scores"])
        masks2 = masks[n_samples : 2 * n_samples].to(data["scores"])
        scores = (
            (data["scores"][:n_samples] >= data["scores"][n_samples : 2 * n_samples])
            .float()
            .to(data["scores"])
        )

        return papers1, masks1, papers2, masks2, scores
