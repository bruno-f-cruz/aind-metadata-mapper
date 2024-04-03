from aind_metadata_mapper.bruker.mri_loader import MRILoader
from pathlib import Path
import json
import os
import unittest
from datetime import datetime

from unittest.mock import MagicMock, patch


from aind_metadata_mapper.bruker.mri_loader import JobSettings, MRIEtl
from aind_data_schema.models.devices import Scanner, ScannerLocation, MagneticStrength
from aind_data_schema.core.mri_session import MRIScan, MriSession, MriScanSequence, ScanType, SubjectPosition
from aind_metadata_mapper.bruker.MRI_ingest.bruker2nifti._metadata import BrukerMetadata


EXAMPLE_MRI_INPUT = "src/aind_metadata_mapper/bruker/MRI_ingest/MRI_files/RawData-2023_07_21/RAW/DL_AI2.kX2"
EXPECTED_MRI_SESSION = "tests/resources/bruker/test_mri_session.json"



class TestMRIWriter(unittest.TestCase):
    """Test methods in SchemaWriter class."""

    @classmethod
    def setUpClass(cls):
        """Load record object and user settings before running tests."""

        with open(EXPECTED_MRI_SESSION, "r") as f:
            expected_session_contents = MriSession(**json.load(f))

        cls.example_job_settings = JobSettings(
            data_path=EXAMPLE_MRI_INPUT,
            output_directory=Path("src/aind_metadata_mapper/bruker/MRI_ingest/output"),
            experimenter_full_name=["Mae"],
            primary_scan_number=7,
            setup_scan_number=1,
            scan_location=ScannerLocation.FRED_HUTCH,
            MagneticStrength=MagneticStrength.MRI_7T,
            notes="test",
        )

        cls.expected_session = expected_session_contents

    
    def test_constructor_from_string(self) -> None:
        """Test constructor from string."""

        job_settings_string = self.example_job_settings.model_dump_json()
        etl0 = MRIEtl(self.example_job_settings)
        etl1 = MRIEtl(job_settings_string)

        self.assertEqual(etl1.job_settings, etl0.job_settings)

    def test_etl(self) -> None:
        """Tests that the extract method returns the correct data."""

        class dummy_input:
            def __init__(self, scan_data, subject_data):
                self.scan_data = scan_data
                self.subject_data = subject_data

        with open(EXAMPLE_MRI_INPUT, "r") as f:
            raw_md_contents = f.read(

        etl = MRIEtl(self.example_job_settings)

        expected_data = BrukerMetadata(EXAMPLE_MRI_INPUT)

        self.assertEqual(extracted_data, expected_data)

        self.assertEqual(extracted_data.metadata.)
