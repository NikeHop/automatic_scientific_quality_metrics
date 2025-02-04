#! /bin/bash

set -e 

mkdir -p model_store

cd model_store

# Download models 
echo "Donwloading the score prediction models"

# Review Score Predictor ICLR Title + Abstract
if [ ! -f "./model_store/iclr_score_prediction.ckpt" ]; then
    gdown 1h0pEOamz7u8x4li5M6UZBfdww_aU7ieY
fi

# Review Score Predictor NeurIPS Title + Abstract
if [ ! -f "./model_store/neurips_score_prediction.ckpt" ]; then
    gdown 1GYfnAhMb-2ee5-zlsePStN8ASi2lUBEc
fi

# Review Pairwise Comparison ICLR Title + Abstract
if [ ! -f "./model_store/iclr_pairwise_comparison.ckpt" ]; then
    gdown 1fmolh5_vzDictgJIZ3RE5fCNvL5H6Rf0
fi

# Review Pairwise Comparison NeurIPS Title + Abstract
if [ ! -f "./model_store/neurips_pairwise_comparison.ckpt" ]; then
    gdown 1mniuos-obvxLnukCW-zyLjSWUndUrr07
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



