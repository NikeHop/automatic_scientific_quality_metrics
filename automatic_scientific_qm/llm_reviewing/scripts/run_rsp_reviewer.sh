#! /bin/bash

set -e 

mkdir -p model_store

cd model_store

# Download models 
echo "Donwloading the score prediction models"

# Review Score Predictor ICLR Title + Abstract
if [ ! -f "./model_store/iclr_score_prediction.ckpt" ]; then
    gdown 1n42AggcFJcoU-LSj0ME1wYCHbSrlLV-G
fi

# Review Score Predictor NeurIPS Title + Abstract
if [ ! -f "./model_store/neurips_score_prediction.ckpt" ]; then
    gdown 1W1yFGXYBTrk4_-XesATYbsLq1jLzeF1H
fi

# Review Pairwise Comparison ICLR Title + Abstract
if [ ! -f "./model_store/iclr_pairwise_comparison.ckpt" ]; then
    gdown 1IJAotMis3XBykWotdl4DItGO9WydzhS6
fi

# Review Pairwise Comparison NeurIPS Title + Abstract
if [ ! -f "./model_store/neurips_pairwise_comparison.ckpt" ]; then
    gdown 1OBcyNVtp6653VTuPoBDx94z_GIU9LYF_ 
fi

cd ..

# Run score prediction models for ICLR and NeurIPS
echo "Run score prediction models for ICLR and NeurIPS"
python run_score_models.py --config ./configs/rsp_review.yaml --dataset "iclr" --checkpoint ./model_store/iclr_score_prediction.ckpt
python run_score_models.py --config ./configs/rsp_review.yaml --dataset "neurips" --checkpoint ./model_store/neurips_score_prediction.ckpt

# Run pairwise comparison models for ICLR and NeurIPS
echo "Run pairwise comparison models for ICLR and NeurIPS"
python run_score_models.py --config ./configs/rsp_comparison_review.yaml --dataset "iclr" --checkpoint ./model_store/iclr_pairwise_comparison.ckpt
python run_score_models.py --config ./configs/rsp_comparison_review.yaml --dataset "neurips" --checkpoint ./model_store/neurips_pairwise_comparison.ckpt



