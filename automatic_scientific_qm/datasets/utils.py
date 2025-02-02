"""
Utilities to complete the OpenReview datasets.
"""

from automatic_scientific_qm.utils.data import Section

class StructuredContent:
    def __init__(self, structured_content: dict) -> None:
        self.structured_content = sections


    def get_section_by_classification(self, section_type: str) -> str:
        for section in self.structured_content.values():
            if section.classification == section_type:
                return section.text
        return ""

    def get_full_text():
        pass




def organize_text(parsed_pdf:dict) -> StructuredContent:
    """
    Organizes the parsed_pdf into a dictionary.
    """
    structured_content = {}
    conclusion_passed = False
    last_sec_num = "Unnumbered"

    if (
        "pdf_parse" not in parsed_pdf
        or "body_text" not in parsed_pdf["pdf_parse"]
    ):
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