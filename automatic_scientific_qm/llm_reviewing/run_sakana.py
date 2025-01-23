"""
Run the SAKANA reviewer for a subsample of the NEURIPS and ICLR 2024 papers
"""

import argparse
import datetime
import logging
import json 
import random 
import os 
import sys
import tqdm 
import yaml


import openai 

from research_assistant.data import Paper 
from research_assistant.sakana_reviewer.review import (
    perform_review,
)
from research_assistant.sakana_reviewer.constants import NEURIPS_FORM, ICLR_FORM



def run_evaluation(config):
    
    if config["dataset"]=="iclr":
        form = ICLR_FORM
    elif config["dataset"]=="neurips":
        form = NEURIPS_FORM
    else:
        raise ValueError(f"Dataset {config['dataset']} not recognized")

    client = openai.Client()
    
    with open(config["venue_file"],"r") as f:
        paperhashes = json.load(f)
    
    if os.path.exists(config["output_file"]):
        with open(config["output_file"],"r") as f:
            paperhash2review = json.load(f)
    else:
        paperhash2review = {}
        
    for paperhash in tqdm.tqdm(paperhashes):

        if paperhash in paperhash2review:
            continue

        print(f"Reviewing paper {paperhash}")
        try:
            paperpath = os.path.join(config["data_dir"],f"{paperhash}.json")
            with open(paperpath) as f:
                paper = json.load(f)
                paper = Paper(**paper)
        except Exception as e:
            print(f"Error loading paper {paperhash}")
            print(e)
            continue
        
        text = paper.get_text(False)

        review = perform_review(
            text,
            config["model"],
            client,
            config["num_reflections"],
            config["num_fs_examples"],
            config["num_reviews_ensemble"],
            config["temperature"],
            review_instruction_form=form
        )

        paperhash2review[paperhash] = review
        print(paperhash2review)
        with open(config["output_file"],"w") as f:
            json.dump(paperhash2review,f,indent=4)

"""
Run the Sakana Reviewer over ICLR 2024 and NeurIPS 2024 test set 
"""

import numpy as np 
import json 
import openai 

from research_assistant.hypothesis_generation.sakana.ai_scientist.llm import (
    get_batch_responses_from_llm,
    get_response_from_llm,
    extract_json_between_markers,
)


from research_assistant.sakana_reviewer.constants import (
    FEW_SHOT_PAPERS_ICLR,
    FEW_SHOT_REVIEWS_ICLR,
    FEW_SHOT_PAPERS_NEURIPS,
    FEW_SHOT_REVIEWS_NEURIPS,
    NEURIPS_FORM,
    REVIEWER_SYSTEM_PROMPT_BASE,
    META_REVIEWER_SYSTEM_PROMPT,
    REVIEWER_REFLECTION_PROMPT
)


def get_review_fewshot_examples(num_fs_examples=1,iclr=False):
    fewshot_prompt = """
    Below are some sample reviews, copied from previous machine learning conferences.
    Note that while each review is formatted differently according to each reviewer's style, the reviews are well-structured and therefore easy to navigate.
    """

    if iclr:
        few_shot_papers = FEW_SHOT_PAPERS_ICLR
        few_shot_reviews = FEW_SHOT_REVIEWS_ICLR
    else:
        few_shot_papers = FEW_SHOT_PAPERS_NEURIPS
        few_shot_reviews = FEW_SHOT_REVIEWS_NEURIPS
    

    for paper_text, review_text in zip(
        few_shot_papers[:num_fs_examples], few_shot_reviews[:num_fs_examples]
    ):
        fewshot_prompt += f"""
        Paper:

        ```
        {paper_text}
        ```

        Review:

        ```
        {review_text}
        ```

        """

    return fewshot_prompt

def perform_review(
    text,
    model,
    client,
    num_reflections=1,
    num_fs_examples=2,
    num_reviews_ensemble=1,
    temperature=0.75,
    msg_history=None,
    return_msg_history=False,
    reviewer_system_prompt=REVIEWER_SYSTEM_PROMPT_BASE,
    review_instruction_form=NEURIPS_FORM,
):
    
    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples,iclr=True)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += f"""
    Here is the paper you are asked to review:
    ```
    {text}
    ```"""


    if num_reviews_ensemble > 1:
        llm_review, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            # Higher temperature to encourage diversity.
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_review):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews, review_instruction_form)

        # take first valid in case meta-reviewer fails
        if review is None:
            review = parsed_reviews[0]

        # Replace numerical scores with the average of the ensemble.
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[1] >= r[score] >= limits[0]:
                    scores.append(r[score])
            review[score] = int(round(np.mean(scores)))

        # Rewrite the message history with the valid one and new aggregated review.
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": f"""
                            THOUGHT:
                            I will start by aggregating the opinions of {num_reviews_ensemble} reviewers that I previously obtained.

                            REVIEW JSON:
                            ```json
                            {json.dumps(review)}
                            ```
                            """,
                }
            ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            # print(f"Relection: {j + 2}/{num_reflections}")
            text, msg_history = get_response_from_llm(
                REVIEWER_REFLECTION_PROMPT,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"

            if "I am done" in text:
                # print(f"Review generation converged after {j + 2} iterations.")
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review
 


def get_meta_review(model, client, temperature, reviews,review_form):
    # Write a meta-review from a set of individual reviews
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
        Review {i + 1}/{len(reviews)}:
        ```
        {json.dumps(r)}
        ```
        """
    base_prompt = review_form + review_text

    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=META_REVIEWER_SYSTEM_PROMPT.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review