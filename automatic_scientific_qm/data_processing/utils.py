"""
Utilities to complete the OpenReview datasets.
"""

import nltk
import torch

from automatic_scientific_qm.utils.data import Section, fuzzy_matching
from automatic_scientific_qm.section_classification.trainer import SectionClassifier
from automatic_scientific_qm.section_classification.utils import (
    SECTION_SYNONYMS,
    LABEL2SECTION,
)


class StructuredContent:
    def __init__(self, structured_content: dict) -> None:
        self.structured_content = structured_content

    def get_section_by_classification(self, section_type: str) -> str:
        for section in self.structured_content.values():
            if section.classification == section_type:
                return section.text
        return ""

    def get_full_text(self, with_appendix: bool = True) -> str:
        text = ""

        paper_no_appendix = {
            key: value
            for key, value in self.structured_content.items()
            if value.sec_num != "appendix"
        }
        paper_appendix = {
            key: value
            for key, value in self.structured_content.items()
            if value.sec_num == "appendix"
        }

        for value in sorted(
            paper_no_appendix.values(),
            key=lambda y: int(y.sec_num) if y.sec_num != "Unnumbered" else float("inf"),
        ):
            text += f"{value.name}: \n {value.text} \n"

        appendix = ""
        for value in paper_appendix.values():
            appendix += value.text

        if with_appendix:
            text = f"Main paper: \n {text} \n Appendix: {appendix} \n"
        else:
            text = f"Main paper: \n {text} \n"

        return text


def parsedpdf2structured_content(parsed_pdf: dict) -> StructuredContent:
    """
    Organizes the parsed_pdf into a StructuredContent object, that maps sections to Section objects.
    """
    structured_content = {}
    conclusion_passed = False
    last_sec_num = "Unnumbered"

    if "pdf_parse" not in parsed_pdf or "body_text" not in parsed_pdf["pdf_parse"]:
        return {}

    for part in parsed_pdf["pdf_parse"]["body_text"]:
        # Update conclusion passed
        if not conclusion_passed and len(structured_content) > 0:
            sec_names = [sec.name for sec in structured_content.values()]
            match, _ = fuzzy_matching("conclusion", sec_names)
            if match != "":
                conclusion_passed = True

        sec_num = part.get("sec_num")
        section = part.get("section")
        section = section.lower() if section is not None else section

        if sec_num is None:
            sec_num = "appendix" if conclusion_passed else last_sec_num
        else:
            last_sec_num = sec_num

        main_sec, _, sub_sec = sec_num.partition(".")

        # Main sec is not present yet
        if main_sec not in structured_content:
            structured_content[main_sec] = Section(
                name=section, sec_num=main_sec, text=part["text"]
            )
            if sub_sec != None:
                structured_content[main_sec].subsections.append(
                    Section(name=section, sec_num=sub_sec, text=part["text"])
                )
        else:
            # Add text to the section
            structured_content[main_sec].text += part["text"]
            if sub_sec != None:
                # Add subsection
                structured_content[main_sec].subsections.append(
                    Section(name=section, sec_num=sub_sec, text=part["text"])
                )

    return StructuredContent(structured_content)


def annotate(
    structured_content: StructuredContent,
    section_classifier: SectionClassifier,
    config: dict,
) -> StructuredContent:
    """
    Annotates the structured_content with the section classifier.
    """

    for key, section in structured_content.structured_content.items():
        # Is section name part of the synonyms, do manual mapping
        section_classified = False
        for key, synonyms in SECTION_SYNONYMS.items():
            if section.name in synonyms:
                section.classification = key
                section_classified = True
                break

        # Otherwise use classifier
        if not section_classified:
            sentences = nltk.tokenize.sent_tokenize(section.text)
            sentences = sentences[: config["max_n_sentences_per_example"]]
            model_inputs = section_classifier.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(config["device"])

            print(model_inputs["input_ids"].shape)
            with torch.no_grad():
                outputs = section_classifier.embedding_model(**model_inputs)
                model_output = outputs.last_hidden_state.mean(dim=1)

            section_classifier_input = {
                "embeddings": model_output.unsqueeze(0),
                "mask": torch.zeros(1, model_output.shape[0]).to(config["device"]),
                "label": 0,
            }
            predicted_labels = section_classifier.predict(section_classifier_input)
            predicted_label = predicted_labels.argmax(dim=1).cpu().item()
            section.classification = LABEL2SECTION[predicted_label]

    return structured_content
