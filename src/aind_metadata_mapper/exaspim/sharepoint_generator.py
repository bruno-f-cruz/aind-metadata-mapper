"""Script to generate a sharepoint-copyable metadata file from the Exaspim Mouse Tracker Spreadsheet"""

import json
from pathlib import Path
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field




class JobSettings(BaseSettings):
    """Class to hold job settings"""

    input_spreadsheet_path: Path
    output_spreadsheet_path: Optional[Path] = Field(None)
    subjects_to_ingest: List[str]


single_sub_raw_headings = [
    "nROID*",
    "roVol*",
    "roSub*",	
    "roLot*",	
    "roGC*",
    "roVolV*",
    "roTite*",
    "roSub*b",
    "roLot*b",
    "roGC*b",
    "roVolV*b",
    "roTite*b",
    "roSub*c",
    "roLot*c",
    "roGC*c",
    "roVolV*c",
    "roTite*c",
    "roSub*d",
    "roLot*d",
    "roGC*d",
    "roVolV*d",
    "roTite*d",
    "roTube*",
    "roBox*"
]

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

    def extract(self, subj_id, idx):
        """Extract data from the input spreadsheet"""
        pass


    def generate_headings(self):
        """Generate sharepoint metadata file"""
        
        headings = []

        for idx, subj_id in enumerate(self.job_settings_model.subjects_to_ingest):
            new_headings = [heading.replace("*", f"{idx+1}") for heading in single_sub_raw_headings]
            headings.extend(new_headings)

        return headings
    
    def pull_data_from_spreadsheet(self):
        """Pull data from spreadsheet"""
        pass