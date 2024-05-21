import unittest

from unittest.mock import MagicMock
from aind_metadata_mapper.utils import sync_utils as sync

class TestGetMetaData(unittest.TestCase):
    def test_get_meta_data(self):
        # Mock sync file data
        mock_sync_file_data = {
            "meta": '{"key1": "value1", "key2": "value2"}'
        }

        # Mock the h5py.File object
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_sync_file_data[key]

        # Call the function to get meta data
        meta_data = sync.get_meta_data(mock_sync_file)

        # Check if the returned meta data matches the expected data
        expected_meta_data = {'key1': 'value1', 'key2': 'value2'}
        self.assertEqual(meta_data, expected_meta_data)

if __name__ == "__main__":
    unittest.main()
