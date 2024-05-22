from aind_metadata_mapper.bruker.mri_loader import MRIEtl, JobSettings
from pathlib import Path
import json
import os
import unittest
from datetime import datetime
import pickle

from unittest.mock import MagicMock, patch


from aind_metadata_mapper.bruker.mri_loader import JobSettings, MRIEtl
from aind_data_schema.components.devices import Scanner, ScannerLocation, MagneticStrength
from aind_data_schema.core.session import MRIScan, Session, MriScanSequence, ScanType, SubjectPosition
from bruker2nifti._metadata import BrukerMetadata


EXAMPLE_MRI_INPUT = "src/aind_metadata_mapper/bruker/MRI_ingest/MRI_files/RawData-2023_07_21/RAW/DL_AI2.kX2"
EXPECTED_MRI_SESSION = "tests/resources/bruker/test_mri_session.json"

TEST_INPUT_SCAN_DATA = "tests/resources/bruker/test_output_scan"
TEST_INPUT_SUBJECT_DATA = "tests/resources/bruker/test_output_subject"
TEST_INPUT_METADATA = "tests/resources/bruker/test_output_metadata"


class TestMRIWriter(unittest.TestCase):
    """Test methods in SchemaWriter class."""

    @classmethod
    def setUpClass(cls):
        """Load record object and user settings before running tests."""

        cls.example_job_settings = JobSettings(
            data_path=EXAMPLE_MRI_INPUT,
            output_directory=Path("src/aind_metadata_mapper/bruker/MRI_ingest/output"),
            experimenter_full_name=["fake mae"],
            primary_scan_number=7,
            setup_scan_number=1,
            scanner_name="fake scanner",
            scan_location=ScannerLocation.FRED_HUTCH,
            magnetic_strength=MagneticStrength.MRI_7T,
            subject_id="fake subject",
            protocol_id="fake protocol",
            iacuc_protocol="fake iacuc",
            notes="test",
        )

    
    def test_constructor_from_string(self) -> None:
        """Test constructor from string."""

        job_settings_string = self.example_job_settings.model_dump_json()
        etl0 = MRIEtl(self.example_job_settings)
        etl1 = MRIEtl(job_settings_string)

        self.assertEqual(etl1.job_settings, etl0.job_settings)

    @patch("aind_metadata_mapper.bruker.mri_loader.MRIEtl._extract")
    def test_etl(self, mock_extract) -> None:
        """Tests that the transform and load methods returns the correct data."""

        with open(TEST_INPUT_METADATA, "rb") as f:
            metadata = pickle.load(f)

        etl = MRIEtl(self.example_job_settings)

        mock_extract.return_value = metadata
        job_response = etl.run_job()

        print("JOB RESPONSE: ", job_response)

        self.assertEqual(job_response.status_code, 200)
