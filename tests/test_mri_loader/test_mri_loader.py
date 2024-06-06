"""Tests methods defined in the mri_loader module"""

import json
import os
import pickle
import unittest
from pathlib import Path
from unittest.mock import patch

from aind_data_schema.components.devices import (
    MagneticStrength,
    ScannerLocation,
)

from aind_metadata_mapper.bruker.mri_loader import JobSettings, MRIEtl

RESOURCES_DIR = (
    Path(os.path.dirname(os.path.realpath(__file__)))
    / ".."
    / "resources"
    / "bruker"
)

EXPECTED_SESSION = RESOURCES_DIR / "expected_session.json"

EXAMPLE_SCAN_DATA_PATH = RESOURCES_DIR / "example_scan_data.pkl"
EXAMPLE_SUBJECT_DATA_PATH = RESOURCES_DIR / "example_subject_data.pkl"


class TestMRIWriter(unittest.TestCase):
    """Test methods in SchemaWriter class."""

    # Set up a MockBrukerMetadata class to test methods without loading entire
    # data set
    class MockBrukerMetadata:
        """Mock BrukerMetadata class to avoid storing large data set. The two
        functions used in the etl job are parse_subject and parse_scans. They
         set the scan_data and subject_data attributes. This Mock class will
          unpickle data in the test resource directory."""

        def __init__(self, *args, **kwargs):
            """Mock class constructor"""
            self.args = args
            self.kwargs = kwargs
            self.scan_data = None
            self.subject_data = None

        def parse_subject(self):
            """Mock parse_subject to return example data"""
            with open(EXAMPLE_SUBJECT_DATA_PATH, "rb") as f:
                example_subject_data = pickle.load(f)
            self.subject_data = example_subject_data

        def parse_scans(self):
            """Mock parse_scans to return example data"""
            with open(EXAMPLE_SCAN_DATA_PATH, "rb") as f:
                example_scan_data = pickle.load(f)
            self.scan_data = example_scan_data

    @classmethod
    def setUpClass(cls):
        """Load record object and user settings before running tests."""

        with open(EXPECTED_SESSION, "r") as f:
            contents = json.load(f)

        cls.expected_session = contents

        cls.example_job_settings = JobSettings(
            data_path="some_data_path",
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

    @patch(
        "aind_metadata_mapper.bruker.mri_loader.BrukerMetadata",
        new=MockBrukerMetadata,
    )
    def test_extract(self):
        """Tests the _extract method"""
        job = MRIEtl(self.example_job_settings)
        bruker_metadata = job._extract()
        self.assertIsNotNone(bruker_metadata.scan_data)
        self.assertIsNotNone(bruker_metadata.subject_data)

    @patch(
        "aind_metadata_mapper.bruker.mri_loader.BrukerMetadata",
        new=MockBrukerMetadata,
    )
    def test_etl(self) -> None:
        """Tests the run_job method."""
        etl = MRIEtl(self.example_job_settings)
        job_response = etl.run_job()
        actual_session = json.loads(job_response.data)
        self.assertEqual(job_response.status_code, 200)
        self.assertEqual(self.expected_session, actual_session)


if __name__ == "__main__":
    unittest.main()
