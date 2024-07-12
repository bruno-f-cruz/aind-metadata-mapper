"""Test the methods in the SharepointGenerator class."""

from pathlib import Path
from aind_metadata_mapper.exaspim.sharepoint_generator import SharePointGenerator, JobSettings
import unittest




RESOURCES_DIR = Path(__file__).parent / "resources" / "exaspim"

INPUT_SPREADSHEET_PATH = RESOURCES_DIR / "example_input_spreadsheet.xlsx"
EXAMPLE_TRANSFORM_PATH = RESOURCES_DIR / "example_transform_output.csv"
EXAMPLE_COMBINE_PATH = RESOURCES_DIR / "example_combine_output.csv"

class TestSharePointGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.job_settings = JobSettings(
            input_spreadsheet_path="Mouse Tracker - Molecular Anatomy.xlsx",
            input_spreadsheet_sheet_name="Mouse Tracker - Molecular Anato",

            subjects_to_ingest=[
                "717442",
                "717443",
                "717444",
            ]
        )
        
    def test_extract(self):
        """Tests the extract method"""
        generator = SharePointGenerator(job_settings=self.job_settings)
        extracted = generator.extract()
        self.assertIsNotNone(extracted)

    def test_transform(self):
        """Tests the transform method"""
        generator = SharePointGenerator(job_settings=self.job_settings)
        extracted = generator._extract()
        transformed = generator._transform(extracted, "717442", 1)

        with open(EXAMPLE_TRANSFORM_PATH, "r") as f:
            expected_transform = f.read()

        self.assertEqual(transformed, expected_transform)

    def test_combine(self):
        """Tests the combine method"""
        generator = SharePointGenerator(job_settings=self.job_settings)
        extracted = generator._extract()
        transformed = []
        for idx, subj_id in enumerate(self.job_settings.subjects_to_ingest):
            transformed.append(generator._transform(extracted, subj_id, idx + 1))

        combined = generator._combine(transformed)

        with open(EXAMPLE_COMBINE_PATH, "r") as f:
            expected_combine = f.read()

        self.assertEqual(combined, expected_combine)

    def test_load(self):
        """Tests the load method"""
        generator = SharePointGenerator(job_settings=self.job_settings)
        extracted = generator._extract()
        transformed = []
        for idx, subj_id in enumerate(self.job_settings.subjects_to_ingest):
            transformed.append(generator._transform(extracted, subj_id, idx + 1))

        combined = generator._combine(transformed)

        generator._load(combined, "output_spreadsheet_path.csv")

        with open("output_spreadsheet_path.csv", "r") as f:
            loaded = f.read()

        self.assertEqual(combined, loaded)

    def test_generate_curr_subj_headings(self):
        """Tests the generate_curr_subj_headings method"""
        generator = SharePointGenerator(job_settings=self.job_settings)
        generated_headings = generator.generate_curr_subj_headings(1)
        expected_headings = [
            "nROID1",
            "roVol1",
            "roSub1a",
            "roLot1a",
            "roGC1a",
            "roVolV1a",
            "roTite1a",
            "roSub1b",
            "roLot1b",
            "roGC1b",
            "roVolV1b",
            "roTite1b",
            "roSub1c",
            "roLot1c",
            "roGC1c",
            "roVolV1c",
            "roTite1c",
            "roSub1d",
            "roLot1d",
            "roGC1d",
            "roVolV1d",
            "roTite1d",
            "roTube1",
            "roBox1",
        ]

        self.assertEqual(generated_headings, expected_headings)

    def test_generate_headings(self):
        """Tests the generate_headings method"""
        generator = SharePointGenerator(job_settings=self.job_settings)
        generated_headings = generator.generate_headings()
        expected_headings = [
            "nROID1",
            "roVol1",
            "roSub1a",
            "roLot1a",
            "roGC1a",
            "roVolV1a",
            "roTite1a",
            "roSub1b",
            "roLot1b",
            "roGC1b",
            "roVolV1b",
            "roTite1b",
            "roSub1c",
            "roLot1c",
            "roGC1c",
            "roVolV1c",
            "roTite1c",
            "roSub1d",
            "roLot1d",
            "roGC1d",
            "roVolV1d",
            "roTite1d",
            "roTube1",
            "roBox1",
            "nROID2",
            "roVol2",
            "roSub2a",
            "roLot2a",
            "roGC2a",
            "roVolV2a",
            "roTite2a",
            "roSub2b",
            "roLot2b",
            "roGC2b",
            "roVolV2b",
            "roTite2b",
            "roSub2c",
            "roLot2c",
            "roGC2c",
            "roVolV2c",
            "roTite2c",
            "roSub2d",
            "roLot2d",
            "roGC2d",
            "roVolV2d",
            "roTite2d",
            "roTube2",
            "roBox2",
            "nROID3",
            "roVol3",
            "roSub3a",
            "roLot3a",
            "roGC3a",
            "roVolV3a",
            "roTite3a",
            "roSub3b",
            "roLot3b",
            "roGC3b",
            "roVolV3b",
            "roTite3b",
            "roSub3c",
            "roLot3c",
            "roGC3c",
            "roVolV3c",
            "roTite3c",
        ]

        self.assertEqual(generated_headings, expected_headings)