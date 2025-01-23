# Code for the paper "Automatic Evaluation Metrics for Artificially Generated Scientific Research"


## Dependencies

## Datasets


## Train Section Classifier

To train the section classifier on sections from the ACL-OCL dataset run from the `./section_classification directory`:

```
python train.py --config ./configs/train_acl_ocl.yaml
```

Running the code for the first time will embed the dataset using [SPECTER2](https://huggingface.co/allenai/specter2), which takes ~2hr.  

## Train Score Predictors

## Run LLM Reviewers  

**Note 1:** 

This section requires that the complete OpenReview dataset is available. 

**Note 2:** 

This section requires an API key for either OpenAI or Anthropic. If you want to use the Anthropic API, store the API key as an environment variable in `ANTHROPIC_API_KEY`. If you want tp use the OpenAI API, store the API key as an environment variable in
`OPENAI_API_KEY`.

We run the following two LLM-reviewers on a subset of ICLR-2024 and NeurIPS-2024 submissions:

(1) Sakana's LLM reviewer \
(2) Paiwise Comparison Reviewer.


## Updates 

## Licenses
