"""
PyTorch Lightning Trainer for the score prediction models.
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn

from automatic_scientific_qm.score_prediction.model import ScoreModel

class ScorePrediction(pl.LightningModule):
    def __init__(self, config:dict)->None:
        super().__init__()

        self.citation_model = ScoreModel(config["model"])
        self.loss = nn.L1Loss()
        self.lr = config["lr"]
        self.optimizer = self.configure_optimizers()
        self.save_hyperparameters()

    def training_step(self,data, data_idx)->torch.Tensor:
        papers, masks, scores, topics, _ = data
        predicted_scores = self.citation_model(papers,masks,topics).squeeze(1)
        loss = self.loss(predicted_scores,scores)
        self.log("training/loss",loss)
        return loss

    def validation_step(self,data, data_idx)->None:
        papers, masks, scores, topics, _ = data
        predicted_scores = self.citation_model(papers,masks,topics).squeeze(1)
        loss = self.loss(predicted_scores,scores)
        self.log("validation/loss",loss)
        
    def predict(self,data)->torch.Tensor:
        papers, masks, topics = data
        predicted_scores = self.citation_model(papers,masks,topics).squeeze(1)
        return predicted_scores

    def configure_optimizers(self)->torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PairwiseComparison(pl.LightningModule):
    def __init__(self, config:dict)->None:
        super().__init__()
        self.citation_model = ScoreModel(config["model"])
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = config["lr"]
        self.optimizer = self.configure_optimizers()
        self.save_hyperparameters()

    def training_step(self,data, data_idx):
        papers1, masks1, topics1, papers2, masks2, topics2, scores = data
        predicted_scores1 = self.citation_model(papers1,masks1,topics1).squeeze(1)
        predicted_scores2 = self.citation_model(papers2,masks2,topics2).squeeze(1)
        score_diff = predicted_scores1-predicted_scores2
        loss = self.loss(score_diff,scores)
        self.log("training/loss",loss)
        acc = ((score_diff>0)==scores).float().mean()
        self.log("training/acc",acc)
        return loss

    def validation_step(self,data, data_idx):
        papers1, masks1, topics1, papers2, masks2, topics2, scores = data
        predicted_scores1 = self.citation_model(papers1,masks1,topics1).squeeze(1)
        predicted_scores2 = self.citation_model(papers2,masks2,topics2).squeeze(1)
        score_diff = predicted_scores1-predicted_scores2
        loss = self.loss(score_diff,scores)
        self.log("validation/loss",loss)
        acc = ((score_diff>0)==scores).float().mean()
        self.log("validation/acc",acc)
        return loss

    def predict(self,data):
        papers1, masks1, topics1, papers2, masks2, topics2, scores = data
        predicted_scores1 = self.citation_model(papers1,masks1,topics1).squeeze(1)
        predicted_scores2 = self.citation_model(papers2,masks2,topics2).squeeze(1)
        return predicted_scores1>=predicted_scores2

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)