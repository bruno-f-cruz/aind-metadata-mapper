from aind_metadata_mapper.bruker.mri_loader import MRILoader
from pathlib import Path
import json
import os
import unittest
from datetime import datetime

from aind_metadata_mapper.bruker.mri_loader import JobSettings, MRIEtl
from aind_data_schema.models.devices import Scanner, ScannerLocation, MagneticStrength





EXAMPLE_MRI_SESSION = "src/aind_metadata_mapper/bruker/MRI_ingest/MRI_files/RawData-2023_07_21/RAW/DL_AI2.kX2"

class TestMRIWriter(unittest.TestCase):
    """Test methods in SchemaWriter class."""

    @classmethod
    def setUpClass(cls):
        """Load record object and user settings before running tests."""


        cls.example_job_settings = JobSettings(
            data_path=EXAMPLE_MRI_SESSION,
            output_directory=Path("src/aind_metadata_mapper/bruker/MRI_ingest/output"),
            experimenter_full_name=["Mae"],
            primary_scan_number=7,
            setup_scan_number=1,
            scan_location=ScannerLocation.FRED_HUTCH,
            MagneticStrength=MagneticStrength.MRI_7T,
            notes="test",
        )
