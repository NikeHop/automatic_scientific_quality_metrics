"""
Evaluation loop for score prediction and pairwise comparison models.
"""
import itertools
import random 

from collections import defaultdict

import pandas as pd 
import tqdm 
import torch
import wandb

def eval(best_model, test_dataloader, config)->None:
    """
    Evaluate the performance of the best_model on the test dataset.

    Args:
        best_model (torch.nn.Module): The trained model to be evaluated.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        config (dict): Configuration parameters for the evaluation.

    Returns:
        None
    """
    best_model.eval()
    if config["data"]["pairwise_comparison"]:
        eval_pairwise_comparison(best_model, test_dataloader, config)
    else:
        eval_regression(best_model, test_dataloader, config)

    
def eval_pairwise_comparison(best_model, test_dataloader, config)->None:
    """
    Evaluate pairwise comparisons between papers using a given model.

    Args:
        best_model (Model): The trained model used for prediction.
        test_dataloader (DataLoader): The dataloader containing the test dataset.
        config (dict): Configuration parameters for the evaluation.

    Returns:
        None
    """

    test_dataset = test_dataloader.dataset
    n = len(test_dataset)
    paper_ids = list(range(n))

    # Subsample the paper-ids to reduce number of computations
    paper_ids = random.sample(paper_ids, min(len(paper_ids), 1000))

    n_comparisons = 0
    n_correct_comparisons = 0
    id2score = {}
    id2wins = defaultdict(int)
    pbar = tqdm(total=len(paper_ids) * (len(paper_ids) - 1) // 2)
    for paper_id_a, paper_id_b in itertools.combinations(paper_ids, 2):
        paper_a, context_a, score_a, std__a = test_dataset[paper_id_a]
        paper_b, context_b, score_b, std_b = test_dataset[paper_id_b]
        context_a = torch.tensor(context_a).float().to(config["device"])
        context_b = torch.tensor(context_b).float().to(config["device"])
        paper_a = torch.tensor(paper_a).float().to(config["device"])
        paper_b = torch.tensor(paper_b).float().to(config["device"])
        masks_a = ~torch.tensor([[1] * paper_a.shape[0]], dtype=torch.long).bool().to(config['device'])
        masks_b = ~torch.tensor([[1] * paper_b.shape[0]], dtype=torch.long).bool().to(config['device'])

        gt = 1 if score_a > score_b else 0
        with torch.no_grad():
            predicted_comparison = best_model.predict((paper_a, masks_a, None, paper_b, masks_b, None))

        n_comparisons += 1
        if predicted_comparison == gt:
            n_correct_comparisons += 1

        if predicted_comparison == 1:
            id2wins[paper_id_a] += 1
        else:
            id2wins[paper_id_b] += 1

        id2score[paper_id_a] = score_a
        id2score[paper_id_b] = score_b
        pbar.update(1)

    # Log Comparison Accuracy
    if n_comparisons > 0:
        wandb.log({"comparison_accuracy": n_correct_comparisons / n_comparisons})

    # Log Spearman Rank Correlation
    df = pd.DataFrame({"score": id2score, "wins": id2wins})
    spearman_corr = df.corr(method="spearman").iloc[0, 1]
    wandb.log({"spearman_rank_corr": spearman_corr})


def eval_regression(best_model, test_dataloader, config):
    """
    Evaluate the regression performance of a given model on a test dataset.

    Args:
        best_model (Model): The trained regression model to evaluate.
        test_dataloader (DataLoader): The data loader for the test dataset.
        config (dict): Configuration parameters for the evaluation.

    Returns:
        None
    """
    id2score = {}
    id2pred_score = defaultdict(float)
    abs_errors = []
    for k, (paper, context, score, score_std) in tqdm(enumerate(test_dataloader.dataset)):
        score = torch.tensor(score).float()
        context = torch.tensor(context).to(config["device"])
        paper = torch.tensor(paper).to(config["device"])
        masks = ~torch.tensor([[1]*paper.shape[0]], dtype=torch.long).bool().to(config['device'])
        
        with torch.no_grad():
            predicted_score = best_model.predict((paper, masks, None))
        
        id2score[k] = score
        id2pred_score[k] = predicted_score
        abs_errors.append(abs(score - predicted_score))
        
    # Log Mean Absolute Error
    if len(abs_errors) > 0:
        mean_absolute_error = torch.tensor(abs_errors).nanmean().item()
        wandb.log({"mean_absolute_error": mean_absolute_error})

    # Log Spearman Rank Correlation
    df = pd.DataFrame({"citation_score": id2score, "predicted_score": id2pred_score})
    spearman_corr = df.corr(method="spearman").iloc[0, 1]
    wandb.log({"spearman_rank_corr": spearman_corr})
    

