import unittest

import numpy as np

from datetime import datetime, timedelta
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

    def test_get_line_labels(self):
        # Mock meta data
        mock_meta_data = {
            "line_labels": ["label1", "label2", "label3"]
        }

        # Mock the sync file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_meta_data[key]

        # Call the function to get line labels
        line_labels = sync.get_line_labels(mock_sync_file)

        # Check if the returned line labels match the expected labels
        expected_line_labels = ["label1", "label2", "label3"]
        self.assertEqual(line_labels, expected_line_labels)

    def test_process_times(self):
        # Mock sync file data
        mock_sync_file_data = {
            "data": np.array([[0], [100], [200], [4294967295], [0], [10000000000]], dtype=np.uint32)
        }

        # Mock the h5py.File object
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_sync_file_data[key]

        # Call the function to process times
        times = sync.process_times(mock_sync_file)

        # Check if the returned times match the expected times
        expected_times = np.array([[0], [100], [200], [4294967295], [4294967296], [10000000000]], dtype=np.int64)
        np.testing.assert_array_equal(times, expected_times)

    def test_get_times(self):
        # Mock sync file data
        mock_sync_file_data = {
            "data": np.array([[0], [100], [200], [4294967295], [0], [10000000000]], dtype=np.uint32)
        }

        # Mock the h5py.File object
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_sync_file_data[key]

        # Call the function to get times
        times = sync.get_times(mock_sync_file)

        # Check if the returned times match the expected times
        expected_times = np.array([[0], [100], [200], [4294967295], [4294967296], [10000000000]], dtype=np.int64)
        np.testing.assert_array_equal(times, expected_times)

    def test_get_start_time(self):
        # Mock meta data
        mock_meta_data = {
            "start_time": "2022-05-18T15:30:00"
        }

        # Mock the sync file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_meta_data[key]

        # Call the function to get start time
        start_time = sync.get_start_time(mock_sync_file)

        # Check if the returned start time matches the expected start time
        expected_start_time = sync.datetime.fromisoformat("2022-05-18T15:30:00")
        self.assertEqual(start_time, expected_start_time)

    def test_get_total_seconds(self):
        # Mock meta data
        mock_meta_data = {
            "total_samples": 10000
        }

        # Mock the sync file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_meta_data[key]

        # Mock get_sample_freq function
        def mock_get_sample_freq(meta_data):
            return 100 # Sample frequency is 100 Hz

        # Replace the original get_sample_freq function with the mock
        with unittest.mock.patch("sync.get_sample_freq", side_effect=mock_get_sample_freq):
            # Call the function to get total seconds
            total_seconds = sync.get_total_seconds(mock_sync_file)

            # Check if the returned total seconds matches the expected value
            expected_total_seconds = 10000 / 100
            self.assertEqual(total_seconds, expected_total_seconds)


    def test_get_stop_time(self):
        # Mock start time
        mock_start_time = datetime(2022, 5, 18, 15, 30, 0)

        # Mock total seconds
        mock_total_seconds = 3600  # For example

        # Mock get_start_time function
        def mock_get_start_time(sync_file):
            return mock_start_time

        # Mock get_total_seconds function
        def mock_get_total_seconds(sync_file):
            return mock_total_seconds

        # Mock the sync file
        mock_sync_file = MagicMock()

        # Replace the original get_start_time and get_total_seconds functions with the mocks
        with unittest.mock.patch("sync.get_start_time", side_effect=mock_get_start_time), \
             unittest.mock.patch("sync.get_total_seconds", side_effect=mock_get_total_seconds):
            # Call the function to get stop time
            stop_time = sync.get_stop_time(mock_sync_file)

            # Check if the returned stop time matches the expected value
            expected_stop_time = mock_start_time + timedelta(seconds=mock_total_seconds)
            self.assertEqual(stop_time, expected_stop_time)

    def test_extract_led_times_rising_edges_found(self):
        # Mock get_edges function to return rising edges
        def mock_get_edges(sync_file, kind, keys, units, permissive):
            return np.array([1, 2, 3])  # Example rising edges

        # Mock the sync file
        mock_sync_file = MagicMock()

        # Replace the original get_edges function with the mock
        with unittest.mock.patch("sync.get_edges", side_effect=mock_get_edges):
            # Call the function to extract LED times
            led_times = sync.extract_led_times(mock_sync_file)

            # Check if the returned LED times match the expected rising edges
            expected_led_times = np.array([1, 2, 3])
            np.testing.assert_array_equal(led_times, expected_led_times)

    def test_extract_led_times_rising_edges_not_found(self):
        # Mock get_edges function to raise a KeyError
        def mock_get_edges(sync_file, kind, keys, units, permissive):
            raise KeyError("Rising edges not found")

        # Mock get_rising_edges function to return rising edges
        def mock_get_rising_edges(sync_file, line, units):
            return np.array([4, 5, 6])  # Example rising edges

        # Mock the sync file
        mock_sync_file = MagicMock()

        # Replace the original get_edges and get_rising_edges functions with the mocks
        with unittest.mock.patch("sync.get_edges", side_effect=mock_get_edges), \
            unittest.mock.patch("sync.get_rising_edges", side_effect=mock_get_rising_edges):
            # Call the function to extract LED times
            led_times = sync.extract_led_times(mock_sync_file)

            # Check if the returned LED times match the expected rising edges from the fallback line
            expected_led_times = np.array([4, 5, 6])
            np.testing.assert_array_equal(led_times, expected_led_times)

if __name__ == "__main__":
    unittest.main()
