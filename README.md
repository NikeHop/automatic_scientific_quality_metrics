# Code for the paper "Automatic Evaluation Metrics for Artificially Generated Scientific Research"

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Dependencies


## Datasets

The datasets used in this project can be found on Huggingface:

* [Section Classification](https://huggingface.co/datasets/nhop/academic-section-classification)

* [Scholarly Document Quality Prediction (SDQP)](https://huggingface.co/datasets/nhop/scientific-quality-score-prediction)

To obtain the full OpenReview dataset with all parsed and section annotated pdfs run from the `./datasets` directory the script:

```
bash complete_openreview_dataset.sh
```


## Train Section Classifier

To train the section classifier on sections from the ACL-OCL dataset run from the `./section_classification directory`:

```
python train.py --config ./configs/train_acl_ocl.yaml
```

Running the code for the first time will embed the dataset using [SPECTER2](https://huggingface.co/allenai/specter2), which takes ~2hr.  

## Train Score Predictors

To train the score prediction models run from the `./score_prediction` directory the following command:

```
python train.py --config ./configs/name_of_config.yaml
```

### Citation Count prediction ACL-OCL
For the citation count prediction models on the ACL-OCL dataset use/modify the config `acl_ocl_citation_prediction.yaml`:


| Parameter | Values | 
|-----------|------|
|data/paper_representation | title_abstract, <br> intro, <br> conclusion, <br> background, <br> method, <br> result_experiment, <br> hypothesis |
|data/context_type| no_context, <br> references, <br> full_paper|


### Score prediction OpenReview

For the score prediction models on the OpenReview dataset use/modify the `openreview_score_prediction.yaml`:

| Parameter | Values | 
|-----------|------|
|data/dataset | openreview-full, <br> openreview-iclr, <br> openreview-neurips 
|data/score_type | avg_citations_per_month, <br> mean_score, <br> mean_impact  |
|data/paper_representation | title_abstract,<br> hypothesis |
|data/context_type| no_context, <br> references, <br> full_paper|


If the code is run for the first time the text will be embedded using [SPECTER2](https://huggingface.co/allenai/specter2), which takes ~2hr. 
## Run LLM Reviewers  

**Note 1:** 

This section requires that the complete OpenReview dataset is available. 

**Note 2:** 

This section requires an API key for either OpenAI or Anthropic. If you want to use the Anthropic API, store the API key as an environment variable in `ANTHROPIC_API_KEY`. If you want to use the OpenAI API, store the API key as an environment variable in
`OPENAI_API_KEY`.

We run the following two LLM-reviewers on a subset of ICLR-2024 and NeurIPS-2024 submissions:

(1) Sakana's LLM reviewer \
(2) Paiwise Comparison Reviewer.


## Citation


## Acknowledgements
