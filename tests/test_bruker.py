from aind_metadata_mapper.bruker.mri_loader import MRILoader
from pathlib import Path
import json
import os
import unittest
from datetime import datetime
from glob import glob

from aind_metadata_mapper.bruker.mri_loader import JobSettings, MRIEtl
from aind_data_schema.models.devices import Scanner, ScannerLocation, MagneticStrength




paths = glob("C:\\Users\\mae.moninghoff\\Documents\GitHub\\aind-data-schema-sphinx\\aind-metadata-mapper\\src\\aind_metadata_mapper\\bruker\\MRI_ingest\\MRI_files\\RawData2023_06_29")


class TestMRIWriter(unittest.TestCase):
    """Test methods in SchemaWriter class."""

    @classmethod
    def setUpClass(cls):
        """Load record object and user settings before running tests."""
        cls.example_job_settings = JobSettings(
            data_path=paths,
            output_directory=Path("C:\\Users\\mae.moninghoff\\Documents\GitHub\\aind-data-schema-sphinx\\aind-metadata-mapper\\src\\aind_metadata_mapper\\bruker\\MRI_ingest\\MRI_files"),
            string_to_parse="test",
            experimenter_full_name=["Mae"],
            primary_scan_number=7,
            setup_scan_number=1,
            scan_location=ScannerLocation.FRED_HUTCH,
            MagneticStrength=MagneticStrength.MRI_7T,
            notes="test",
        )



for path in paths:
    loader = MRILoader(path)
scan7 = loader.make_model_from_scan('5', '3D Scan', True)
print(scan7)

session = loader.load_mri_session(["Mae"], "7", "1", ScannerLocation.FRED_HUTCH, MagneticStrength.MRI_7T)


session.write_standard_file(output_directory=Path("./output"), prefix=Path("test"))