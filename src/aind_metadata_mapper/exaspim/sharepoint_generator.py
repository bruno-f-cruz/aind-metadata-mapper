"""Script to generate a sharepoint-copyable metadata file from the Exaspim Mouse Tracker Spreadsheet"""

import json
from pathlib import Path
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field

import pandas as pd





class JobSettings(BaseSettings):
    """Class to hold job settings"""

    input_spreadsheet_path: Path
    input_spreadsheet_sheet_name: str
    output_spreadsheet_path: Optional[Path] = Field(None)
    subjects_to_ingest: List[str]


single_sub_raw_headings = [
    "nROID#",
    "roVol#",
    "roSub#",	
    "roLot#",	
    "roGC#",
    "roVolV#",
    "roTite#",
    "roSub#b",
    "roLot#b",
    "roGC#b",
    "roVolV#b",
    "roTite#b",
    "roSub#c",
    "roLot#c",
    "roGC#c",
    "roVolV#c",
    "roTite#c",
    "roSub#d",
    "roLot#d",
    "roGC#d",
    "roVolV#d",
    "roTite#d",
    "roTube#",
    "roBox#"
]

relevant_sub_headings = [
    "nROID#",
    "roVol#",
]

relevant_inj_headings = [
    "roSub#*",
    "roLot#*",
    "roGC#*",
    "roTite#*",
]

headings_map = {
    "nROID": "Mouse ID",
    "roVol#": "Virus Mix Total Volume injected RO (uL)",
    "roSub#*": "Virus*",
    "roLot#*": "Virus* ID",
    "roGC#*": "Virus* Dose (GC)",
    "roTite#*": "Virus* Effective Titer (GC/mL)",
}

subheadings_map = {
    "a": "1",
    "b": "2",
    "c": "3",
}

def replace_number(str, idx):
    return str.replace("#", f"{idx}")

def replace_letter(str, letter):
    return str.replace("*", f"{letter}")

def replace_heading_counters(str, idx, letter):
    return replace_letter(replace_number(str, idx), letter)


class SharePointGenerator:
    """Class to generate sharepoint metadata file from Exaspim Mouse Tracker Spreadsheet"""

    def __init__(self, job_settings: Union[JobSettings, str]):
        """
        Class constructor for Base etl class.
        Parameters
        ----------
        job_settings: Union[JobSettings, str]
          Variables for a particular session
        """

        if isinstance(job_settings, str):
            self.job_settings_model = JobSettings.model_validate_json(job_settings)
        else:
            self.job_settings_model = job_settings

    def _extract(self):
        """Extract the input spreadsheet"""

        materials_sheet = pd.read_excel(
            self.job_settings_model.input_spreadsheet_path,
            sheet_name=self.job_settings_model.input_spreadsheet_sheet_name,
            header=[0],
            converters={}
        )

        return materials_sheet


    def _transform(self, materials_sheet, subj_id, subj_idx):
        """Extract data from the input spreadsheet"""

        subj_row = materials_sheet.loc[materials_sheet["Mouse ID"] == int(subj_id)]

        output = {}
        
        for header in relevant_sub_headings:
            input_sheet_header = replace_number(headings_map[header], subj_idx)
            output[header] = subj_row[input_sheet_header]

        for letter, number in subheadings_map.items():
            for header in relevant_inj_headings:
                sheet_header = replace_heading_counters(headings_map[header], idx, letter)
                
                output[
                    replace_heading_counters(
                        header,
                        idx,
                        letter
                    )
                ] = subj_row[
                    sheet_header + subheadings_map[subheading]
                ].values[0]

        return output


    def generate_headings(self):
        """Generate sharepoint metadata file"""
        
        headings = []

        for idx, subj_id in enumerate(self.job_settings_model.subjects_to_ingest):
            new_headings = [heading.replace("#", f"{idx+1}") for heading in single_sub_raw_headings]
            headings.extend(new_headings)

        return headings
    
    def pull_data_from_spreadsheet(self):
        """Pull data from spreadsheet"""
        pass