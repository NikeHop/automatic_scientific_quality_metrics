"""
Download the OpenReview datasets from HuggingFace and add the parsed and annotated pdfs to the dataset, save locally.
"""

import argparse
import logging
import os
import time

import tqdm 

from urllib.error import HTTPError
from urllib.request import urlretrieve

from datasets import load_dataset

from doc2json.grobid2json.grobid.grobid_client import (
    GrobidClient,
)
from doc2json.grobid2json.tei_to_json import (
    convert_tei_xml_file_to_s2orc_json,
)

from automatic_scientific_qm.section_classification.trainer import SectionClassifier
from automatic_scientific_qm.datasets.utils import parsedpdf2structured_content, annotate


def complete_openreview_dataset(config: dict) -> None:

    # Setup GROBID client for pdf parsing
    grobid_client = GrobidClient()
    grobid_client.max_workers = config["max_workers"]

    # Prepare PDF directory
    pdf_directory = os.path.join(config["save_directory"], "pdfs")
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
    
    # Prepare XMLs directory
    xml_directory = os.path.join(config["save_directory"], "xmls")
    if not os.path.exists(xml_directory):
        os.makedirs(xml_directory)

    # Load data and model 
    dataset = load_dataset("nhop/scientific-quality-score-prediction", config["dataset_name"])
    section_classifier = SectionClassifier.load_from_checkpoint(
        "section_classifier_openreview.ckpt", map_location=config["device"]
    )

    # Iterate over samples in the dataset
    for split in ["train", "validation", "test"]:
       
        for sample in tqdm.tqdm(dataset[split]):
            # Obtain the pdf
            pdf_id = sample["openreview_submission_id"]
    
            try:
                pdf_filename = os.path.join(pdf_directory, f"{sample['paperhash']}.pdf")
                if not os.path.exists(filename):
                    url = f"https://openreview.net/pdf?id={pdf_id}"
                    urlretrieve(url, pdf_filename)
                
            except HTTPError as e:                
                logging.warning(
                    f"Could not download the pdf for submission {sample["paperhash"] (pdf_id)}."
                )
                logging.warning(e)
                filename = None
            

            if filename!=None:
                # PDF -> XML
                grobid_client.process_batch([filename], xml_directory, "processFulltextDocument")
                
                # XML -> JSON
                xml_filename = os.path.join(xml_directory, f"{sample['paperhash']}.xml")
                parsed_pdf = convert_tei_xml_file_to_s2orc_json(xml_filename)
                parsed_pdf = parsed_pdf.release_json()

                # JSON -> structured content
                structured_content = parsedpdf2structured_content(parsed_pdf)
                structured_content = annotate(structured_content)

                # Annotate the pdf with the section classifier
                introduction = structured_content.get_section_by_classification("introduction")
                background = structured_content.get_section_by_classification("background")
                methodology = structured_content.get_section_by_classification("methodology")
                experiments_results = structured_content.get_section_by_classification("experiments_results")
                conclusion = structured_content.get_section_by_classification("conclusion")
                full_text = structured_content.get_full_text()

            else:
                introduction = None
                background = None
                methodology = None
                experiments_results = None
                conclusion = None
                full_text = None


        sample["introduction"] = introduction
        sample["background"] = background
        sample["methodology"] = methodology
        sample["experiments_results"] = experiments_results
        sample["conclusion"] = conclusion
        sample["full_text"] = full_text

        # Add the samples to the dataset
    
    dataset.save_to_disk(config["save_directory"])
                        
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--openreview_dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    complete_openreview_dataset(args.openreview_dataset, args.output_path)
