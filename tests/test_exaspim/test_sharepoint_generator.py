"""Test the methods in the SharepointGenerator class."""

from pathlib import Path
from aind_metadata_mapper.exaspim.sharepoint_generator import SharePointGenerator

RESOURCES_DIR = Path(__file__).parent / "resources" / "exaspim"

INPUT_SPREADSHEET_PATH = RESOURCES_DIR / "example_input_spreadsheet.xlsx"
