"""File for testing the MRI Loader package"""

import pickle
import unittest
from unittest.mock import patch

from aind_data_schema.components.devices import (
    MagneticStrength,
    ScannerLocation,
)

from aind_metadata_mapper.bruker.mri_loader import JobSettings, MRIEtl

EXAMPLE_MRI_INPUT = (
    "src/aind_metadata_mapper/bruker/MRI_ingest/"
    "MRI_files/RawData-2023_07_21/RAW/DL_AI2.kX2"
)
EXPECTED_MRI_SESSION = "tests/resources/bruker/test_mri_session.json"

TEST_INPUT_SCAN_DATA = "tests/resources/bruker/test_output_scan"
TEST_INPUT_SUBJECT_DATA = "tests/resources/bruker/test_output_subject"
TEST_INPUT_METADATA = "tests/resources/bruker/test_output_metadata_string.pickle"


class TestMRIWriter(unittest.TestCase):
    """Test methods in SchemaWriter class."""

    @classmethod
    def setUpClass(cls):
        """Load record object and user settings before running tests."""

        cls.example_job_settings = JobSettings(
            data_path=EXAMPLE_MRI_INPUT,
            experimenter_full_name=["fake mae"],
            primary_scan_number=7,
            setup_scan_number=1,
            scanner_name="fake scanner",
            session_type="3D MRI Scan",
            scan_location=ScannerLocation.FRED_HUTCH,
            magnetic_strength=MagneticStrength.MRI_7T,
            subject_id="fake subject",
            iacuc_protocol="fake iacuc",
            session_notes="test",
        )

    def test_constructor_from_string(self) -> None:
        """Test constructor from string."""

        job_settings_string = self.example_job_settings.model_dump_json()
        etl0 = MRIEtl(self.example_job_settings)
        etl1 = MRIEtl(job_settings_string)

        self.assertEqual(etl1.job_settings, etl0.job_settings)

    @patch("aind_metadata_mapper.bruker.mri_loader.MRIEtl._extract")
    def test_etl(self, mock_extract) -> None:
        """Tests that ETL methods return the correct data."""

        with open(TEST_INPUT_METADATA, "rb") as f:
            metadata = pickle.load(f)

        etl = MRIEtl(self.example_job_settings)

        mock_extract.return_value = metadata
        job_response = etl.run_job()

        self.assertEqual(job_response.status_code, 200)
