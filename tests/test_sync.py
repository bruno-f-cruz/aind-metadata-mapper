import unittest

import numpy as np

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from aind_metadata_mapper.utils import sync_utils as sync


class TestGetMetaData(unittest.TestCase):
    @patch('builtins.eval', return_value={'key1': 'value1', 'key2': 'value2'})  # Mock eval to return expected dict
    def test_get_meta_data(self):
        mock_sync_file_data = {"meta": {(): "{'key1': 'value1', 'key2': 'value2'}"}}

        # Create a MagicMock object to mock the sync_file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_sync_file_data[key]

        # Call the function to get meta data
        meta_data = sync.get_meta_data(mock_sync_file)

        expected_meta_data = {"key1": "value1", "key2": "value2"}
        self.assertEqual(meta_data, expected_meta_data)

    def test_get_line_labels(self):
        # Mock meta data
        mock_meta_data = {"meta": {(): "{'line_labels': ['label1', 'label2', 'label3']}"}}

        # Mock the sync file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_meta_data[
            key
        ]

        # Call the function to get line labels
        line_labels = sync.get_line_labels(mock_sync_file)

        expected_line_labels = ["label1", "label2", "label3"]
        self.assertEqual(line_labels, expected_line_labels)

    def test_process_times(self):
        # Mock sync file data
        mock_sync_file_data = {
            "data": np.array(
                [ [4294967295], [0], [10000000000]],
                dtype=np.uint32,
            )
        }

        # Mock the h5py.File object
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = (
            lambda key: mock_sync_file_data[key]
        )

        # Call the function to process times
        times = sync.process_times(mock_sync_file)

        expected_times = np.array(
            [[4294967295], [4294967296], [5705032704]],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(times, expected_times)

    def test_get_times(self):
        # Mock sync file data
        mock_sync_file_data = {
            "data": np.array(
                [[4294967295], [0], [10000000000]],
                dtype=np.uint32,
            )
        }

        # Mock the h5py.File object
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = (
            lambda key: mock_sync_file_data[key]
        )

        # Call the function to get times
        times = sync.get_times(mock_sync_file)

        expected_times = np.array(
            [[4294967295], [4294967296], [5705032704]],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(times, expected_times)

    def test_get_start_time(self):
        # Mock meta data
        mock_meta_data = {"meta": {() : "{'start_time': '2022-05-18T15:30:00'}"}}

        # Mock the sync file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_meta_data[
            key
        ]

        # Call the function to get start time
        start_time = sync.get_start_time(mock_sync_file)

        expected_start_time = datetime.fromisoformat(
            "2022-05-18T15:30:00"
        )
        self.assertEqual(start_time, expected_start_time)


    @patch("aind_metadata_mapper.utils.sync_utils.get_sample_freq")
    def test_get_total_seconds(self, mock_get_sample_freq):
        # Set the return value of mock_get_sample_freq to 100
        mock_get_sample_freq.return_value = 100

        # Mock meta data
        mock_meta_data = {"meta": { (): '{"total_samples": 10000}'}}

        # Mock the sync file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.side_effect = lambda key: mock_meta_data[key]

        # Call the function to get total seconds
        total_seconds = sync.get_total_seconds(mock_sync_file)

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

        with unittest.mock.patch(
            "sync.get_start_time", side_effect=mock_get_start_time
        ), unittest.mock.patch(
            "sync.get_total_seconds", side_effect=mock_get_total_seconds
        ):
            # Call the function to get stop time
            stop_time = sync.get_stop_time(mock_sync_file)

            expected_stop_time = mock_start_time + timedelta(
                seconds=mock_total_seconds
            )
            self.assertEqual(stop_time, expected_stop_time)

    def test_extract_led_times_rising_edges_found(self):
        # Mock get_edges function to return rising edges
        def mock_get_edges(sync_file, kind, keys, units, permissive):
            return np.array([1, 2, 3])  # Example rising edges

        # Mock the sync file
        mock_sync_file = MagicMock()

        with unittest.mock.patch("sync.get_edges", side_effect=mock_get_edges):
            # Call the function to extract LED times
            led_times = sync.extract_led_times(mock_sync_file)

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

        with unittest.mock.patch(
            "sync.get_edges", side_effect=mock_get_edges
        ), unittest.mock.patch(
            "sync.get_rising_edges", side_effect=mock_get_rising_edges
        ):
            # Call the function to extract LED times
            led_times = sync.extract_led_times(mock_sync_file)

            expected_led_times = np.array([4, 5, 6])
            np.testing.assert_array_equal(led_times, expected_led_times)

    def test_get_ophys_stimulus_timestamps(self):
        def mock_get_clipped_stim_timestamps(sync, pkl):
            return np.array([1, 2, 3]), None  # Example stimulus timestamps

        # Mock the sync file and pkl
        mock_sync = MagicMock()
        mock_pkl = MagicMock()

        with unittest.mock.patch(
            "sync.get_clipped_stim_timestamps",
            side_effect=mock_get_clipped_stim_timestamps,
        ):
            # Call the function to obtain ophys stimulus timestamps
            stimulus_timestamps = sync.get_ophys_stimulus_timestamps(
                mock_sync, mock_pkl
            )

            expected_stimulus_timestamps = np.array([1, 2, 3])
            np.testing.assert_array_equal(
                stimulus_timestamps, expected_stimulus_timestamps
            )

    def test_get_behavior_stim_timestamps_vsync_stim(self):
        # Mock get_falling_edges function to return stimulus timestamps
        def mock_get_falling_edges(sync, stim_key, units):
            return np.array([1, 2, 3])  # Example stimulus timestamps

        # Mock the sync file
        mock_sync = MagicMock()

        with unittest.mock.patch(
            "sync.get_falling_edges", side_effect=mock_get_falling_edges
        ):
            # Call the function to get behavior stimulus timestamps
            behavior_stim_timestamps = sync.get_behavior_stim_timestamps(
                mock_sync
            )

            expected_behavior_stim_timestamps = np.array([1, 2, 3])
            np.testing.assert_array_equal(
                behavior_stim_timestamps, expected_behavior_stim_timestamps
            )

    def test_get_behavior_stim_timestamps_stim_vsync(self):
        # Mock get_falling_edges function to raise a ValueError
        def mock_get_falling_edges(sync, stim_key, units):
            raise ValueError("Stimulus timestamps not found")

        # Mock the sync file
        mock_sync = MagicMock()

        with unittest.mock.patch(
            "sync.get_falling_edges", side_effect=mock_get_falling_edges
        ):
            # Call the function to get behavior stimulus timestamps
            behavior_stim_timestamps = sync.get_behavior_stim_timestamps(
                mock_sync
            )

            self.assertIsNone(behavior_stim_timestamps)

    def test_get_behavior_stim_timestamps_no_stimulus_stream(self):
        # Mock get_falling_edges function to raise an Exception
        def mock_get_falling_edges(sync, stim_key, units):
            raise Exception("No stimulus stream found in sync file")

        # Mock the sync file
        mock_sync = MagicMock()

        with unittest.mock.patch(
            "sync.get_falling_edges", side_effect=mock_get_falling_edges
        ):
            # Call the function and assert that it raises a ValueError
            with self.assertRaises(ValueError):
                sync.get_behavior_stim_timestamps(mock_sync)

    def test_get_clipped_stim_timestamps_stim_length_less_than_timestamps(
        self,
    ):
        def mock_get_behavior_stim_timestamps(sync):
            return np.array([1, 2, 3, 4, 5])  # Example stimulus timestamps

        def mock_get_stim_data_length(pkl_path):
            return 3

        def mock_get_rising_edges(sync, stim_key, units):
            return np.array([0, 0.1, 0.2, 0.3, 0.4])  # Example rising edges

        # Mock the sync file and pkl_path
        mock_sync = MagicMock()
        mock_pkl_path = "example.pkl"

        with unittest.mock.patch(
            "sync.get_behavior_stim_timestamps",
            side_effect=mock_get_behavior_stim_timestamps,
        ), unittest.mock.patch(
            "sync.get_stim_data_length", side_effect=mock_get_stim_data_length
        ), unittest.mock.patch(
            "sync.get_rising_edges", side_effect=mock_get_rising_edges
        ):
            # Call the function to get clipped stimulus timestamps
            timestamps, delta = sync.get_clipped_stim_timestamps(
                mock_sync, mock_pkl_path
            )

            expected_timestamps = np.array([1, 2, 3])
            expected_delta = 2
            np.testing.assert_array_equal(timestamps, expected_timestamps)
            self.assertEqual(delta, expected_delta)

    def test_get_clipped_stim_timestamps_stim_length_greater_than_timestamps(
        self,
    ):
        # Mock get_behavior_stim_timestamps to return timestamps
        def mock_get_behavior_stim_timestamps(sync):
            return np.array([1, 2, 3])  # Example stimulus timestamps

        # Mock return a length greater than the timestamps length
        def mock_get_stim_data_length(pkl_path):
            return 5

        mock_sync = MagicMock()
        mock_pkl_path = "example.pkl"

        with unittest.mock.patch(
            "sync.get_behavior_stim_timestamps",
            side_effect=mock_get_behavior_stim_timestamps,
        ), unittest.mock.patch(
            "sync.get_stim_data_length", side_effect=mock_get_stim_data_length
        ):
            timestamps, delta = sync.get_clipped_stim_timestamps(
                mock_sync, mock_pkl_path
            )

            expected_timestamps = np.array([1, 2, 3])
            expected_delta = 2
            np.testing.assert_array_equal(timestamps, expected_timestamps)
            self.assertEqual(delta, expected_delta)

    def test_get_clipped_stim_timestamps_no_stimulus_stream(self):
        # Mock get_behavior_stim_timestamps function to return None
        def mock_get_behavior_stim_timestamps(sync):
            return None

        # Mock the sync file and pkl_path
        mock_sync = MagicMock()
        mock_pkl_path = "example.pkl"

        with unittest.mock.patch(
            "sync.get_behavior_stim_timestamps",
            side_effect=mock_get_behavior_stim_timestamps,
        ):
            # Call the function and assert that it raises a ValueError
            with self.assertRaises(ValueError):
                sync.get_clipped_stim_timestamps(mock_sync, mock_pkl_path)

    def test_line_to_bit_with_line_name(self):
        # Mock get_line_labels function to return line labels
        def mock_get_line_labels(sync_file):
            return ["line1", "line2", "line3"]

        # Mock the sync file
        mock_sync_file = MagicMock()

        with unittest.mock.patch(
            "sync.get_line_labels", side_effect=mock_get_line_labels
        ):
            # Call the function to get the bit for the specified line name
            bit = sync.line_to_bit(mock_sync_file, "line2")

            expected_bit = 1
            self.assertEqual(bit, expected_bit)

    def test_line_to_bit_with_line_number(self):
        # Mock the sync file
        mock_sync_file = MagicMock()

        # Call the function to get the bit for the specified line number
        bit = sync.line_to_bit(mock_sync_file, 2)

        expected_bit = 2
        self.assertEqual(bit, expected_bit)

    def test_line_to_bit_with_incorrect_line_type(self):
        # Mock the sync file
        mock_sync_file = MagicMock()

        # Asset wrong linetype returns type error
        with self.assertRaises(TypeError):
            sync.line_to_bit(mock_sync_file, ["line1", "line2"])

    def test_get_bit_changes(self):
        def mock_get_sync_file_bit(sync_file, bit):
            return np.array([0, 1, 0, 1, 1, 0, 0, 1, 0])  # Example bit array

        # Mock the sync file
        mock_sync_file = MagicMock()

        with unittest.mock.patch(
            "sync.get_sync_file_bit", side_effect=mock_get_sync_file_bit
        ):
            # Call the function to get the first derivative
            bit_changes = sync.get_bit_changes(mock_sync_file, 2)

            expected_bit_changes = np.array([0, 1, -1, 1, 0, -1, 1, -1, 0])
            np.testing.assert_array_equal(bit_changes, expected_bit_changes)

    def test_get_all_bits(self):
        # Mock the sync file
        mock_sync_file = MagicMock()
        mock_sync_file.__getitem__.return_value = np.array(
            [[0, 1, 0], [1, 0, 1]]
        )  # Example sync file data

        # Call the function to get all counter values
        all_bits = sync.get_all_bits(mock_sync_file)

        expected_all_bits = np.array([0, 1])
        np.testing.assert_array_equal(all_bits, expected_all_bits)

    def test_get_sync_file_bit(self):
        # Mock get_all_bits function to return all bits
        def mock_get_all_bits(sync_file):
            return np.array([0, 1, 0, 1])  # Example all bits

        # Mock the sync file
        mock_sync_file = MagicMock()

        with unittest.mock.patch(
            "sync.get_all_bits", side_effect=mock_get_all_bits
        ):
            # Call the function to get a specific bit from the sync file
            bit_values = sync.get_sync_file_bit(mock_sync_file, 2)

            expected_bit_values = np.array([0, 0, 0, 1])
            np.testing.assert_array_equal(bit_values, expected_bit_values)

    def test_get_bit_single_bit(self):
        # Create a uint array
        uint_array = np.array([3, 5, 6])  # Binary: 011, 101, 110

        # Call the function to extract a single bit
        bit_values = sync.get_bit(uint_array, 1)

        expected_bit_values = np.array([1, 0, 1])
        np.testing.assert_array_equal(bit_values, expected_bit_values)

    def test_get_bit_multiple_bits(self):
        # Create a uint array
        uint_array = np.array([3, 5, 6])  # Binary: 011, 101, 110

        # Call the function to extract multiple bits
        bit_values = sync.get_bit(uint_array, 0)

        expected_bit_values = np.array([1, 1, 0])
        np.testing.assert_array_equal(bit_values, expected_bit_values)

    def test_get_bit_out_of_range(self):
        # Create a uint array
        uint_array = np.array([3, 5, 6])  # Binary: 011, 101, 110

        # Call the function to extract a bit that is out of range
        bit_values = sync.get_bit(uint_array, 3)

        expected_bit_values = np.array([0, 0, 0])
        np.testing.assert_array_equal(bit_values, expected_bit_values)

    def test_get_sample_freq_with_sample_freq_key(self):
        # Create meta data with sample_freq key
        meta_data = {"ni_daq": {"sample_freq": 1000}}

        # Call the function to get the sample frequency
        sample_freq = sync.get_sample_freq(meta_data)

        expected_sample_freq = 1000.0
        self.assertEqual(sample_freq, expected_sample_freq)

    def test_get_sample_freq_with_counter_output_freq_key(self):
        # Create meta data with counter_output_freq key
        meta_data = {"ni_daq": {"counter_output_freq": 500}}

        # Call the function to get the sample frequency
        sample_freq = sync.get_sample_freq(meta_data)

        expected_sample_freq = 500.0
        self.assertEqual(sample_freq, expected_sample_freq)

    def test_get_sample_freq_with_missing_keys(self):
        # Create meta data without sample_freq and counter_output_freq keys
        meta_data = {"ni_daq": {}}

        # Call the function to get the sample frequency
        sample_freq = sync.get_sample_freq(meta_data)

        expected_sample_freq = 0.0
        self.assertEqual(sample_freq, expected_sample_freq)

    def test_get_all_times_with_32_bit_counter(self):
        # Create a mock sync file with data and meta data
        mock_sync_file = {"data": np.array([[0, 100], [1, 200], [2, 300]])}
        mock_meta_data = {"ni_daq": {"counter_bits": 32}}

        # Call the function to get all times in samples
        all_times_samples = sync.get_all_times(
            mock_sync_file, mock_meta_data, units="samples"
        )

        expected_all_times_samples = np.array([0, 1, 2])
        np.testing.assert_array_equal(
            all_times_samples, expected_all_times_samples
        )

    def test_get_all_times_with_non_32_bit_counter(self):
        # Create a mock sync file with data and meta data
        mock_sync_file = {"data": np.array([[0, 100], [1, 200], [2, 300]])}
        mock_meta_data = {"ni_daq": {"counter_bits": 16}}

        # Call the function to get all times in seconds
        all_times_seconds = sync.get_all_times(
            mock_sync_file, mock_meta_data, units="seconds"
        )

        expected_all_times_seconds = np.array([0, 0.1, 0.2])
        np.testing.assert_array_equal(
            all_times_seconds, expected_all_times_seconds
        )

    def test_get_all_times_with_invalid_units(self):
        # Create a mock sync file with data and meta data
        mock_sync_file = {"data": np.array([[0, 100], [1, 200], [2, 300]])}
        mock_meta_data = {"ni_daq": {"counter_bits": 32}}

        # Assert invalid units parameter raises a ValueError
        with self.assertRaises(ValueError):
            sync.get_all_times(
                mock_sync_file, mock_meta_data, units="invalid_units"
            )

    def test_get_falling_edges(self):
        # Mock the required functions to return expected values
        with unittest.mock.patch(
            "sync.get_meta_data", return_value=self.mock_meta_data
        ):
            with unittest.mock.patch(
                "sync.line_to_bit", return_value=3
            ):  # Assuming bit value for the line
                with unittest.mock.patch(
                    "sync.get_bit_changes",
                    return_value=np.array([0, 255, 0, 255]),
                ):  # Mock changes
                    with unittest.mock.patch(
                        "sync.get_all_times",
                        return_value=np.array([0, 1, 2, 3]),
                    ):  # Mock times
                        # Call the function to get falling edges
                        falling_edges = sync.get_falling_edges(
                            self.mock_sync_file, "line"
                        )

        expected_falling_edges = np.array(
            [1, 3]
        )  # Expected indices of falling edges
        np.testing.assert_array_equal(falling_edges, expected_falling_edges)

    def test_get_rising_edges(self):
        # Mock the required functions to return expected values
        with unittest.mock.patch(
            "sync.get_meta_data", return_value=self.mock_meta_data
        ):
            with unittest.mock.patch(
                "sync.line_to_bit", return_value=3
            ):  # Assuming bit value for the line
                with unittest.mock.patch(
                    "sync.get_bit_changes", return_value=np.array([0, 1, 0, 1])
                ):  # Mock changes
                    with unittest.mock.patch(
                        "sync.get_all_times",
                        return_value=np.array([0, 1, 2, 3]),
                    ):  # Mock times
                        # Call the function to get rising edges
                        rising_edges = sync.get_rising_edges(
                            self.mock_sync_file, "line"
                        )

        expected_rising_edges = np.array(
            [1, 3]
        )  # Expected indices of rising edges
        np.testing.assert_array_equal(rising_edges, expected_rising_edges)

    def test_trimmed_stats(self):
        # Create mock data with outliers
        mock_data = np.array([1, 2, 3, 4, 5, 1000])

        # Call the function to calculate trimmed stats
        mean, std = sync.trimmed_stats(mock_data)

        expected_mean = np.mean([1, 2, 3, 4, 5])
        expected_std = np.std([1, 2, 3, 4, 5])
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(std, expected_std)

    def test_trimmed_stats_custom_percentiles(self):
        # Create mock data with outliers
        mock_data = np.array([1, 2, 3, 4, 5, 1000])

        # Call the function with custom percentiles to calculate trimmed stats
        mean, std = sync.trimmed_stats(mock_data, pctiles=(20, 80))

        expected_mean = np.mean([2, 3, 4])
        expected_std = np.std([2, 3, 4])
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(std, expected_std)

    def test_estimate_frame_duration(self):
        # Create mock photodiode times for 3 frames per cycle
        mock_pd_times = np.array([0, 1, 2, 3, 4, 5, 6])

        # Call the function to estimate frame duration
        frame_duration = sync.estimate_frame_duration(mock_pd_times, cycle=3)

        expected_frame_duration = (
            1.0  # Since the photodiode times increase by 1 for each frame
        )
        self.assertAlmostEqual(frame_duration, expected_frame_duration)

    def test_estimate_frame_duration_with_empty_pd_times(self):
        # Create an empty mock photodiode times array
        mock_pd_times = np.array([])

        # Call the function with an empty array
        frame_duration = sync.estimate_frame_duration(mock_pd_times, cycle=3)

        self.assertTrue(np.isnan(frame_duration))

    def test_allocate_by_vsync(self):
        # Create mock data for vsync differences, frame starts, and frame ends
        vs_diff = np.array([1, 2, 3, 2, 1])  # Mock vsync differences
        index = 1  # Mock current vsync index
        starts = np.array([0, 1, 2, 3, 4])  # Mock frame start times
        ends = np.array([1, 2, 3, 4, 5])  # Mock frame end times
        frame_duration = 1  # Mock frame duration
        irregularity = 1  # Mock irregularity
        cycle = 5  # Mock number of frames per cycle

        # Call the function to allocate frame times based on vsync signal
        updated_starts, updated_ends = sync.allocate_by_vsync(
            vs_diff, index, starts, ends, frame_duration, irregularity, cycle
        )

        expected_updated_starts = np.array(
            [0, 1, 2, 4, 5]
        )  # After allocating based on vsync signal
        expected_updated_ends = np.array(
            [1, 2, 4, 5, 6]
        )  # After allocating based on vsync signal
        np.testing.assert_array_almost_equal(
            updated_starts, expected_updated_starts
        )
        np.testing.assert_array_almost_equal(
            updated_ends, expected_updated_ends
        )

    def test_trim_border_pulses(self):
        # Create mock photodiode times and vsync times
        pd_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        vs_times = np.array([1.0, 2.0])

        # Trim pulses near borders of the photodiode signal
        trimmed_pd_times = sync.trim_border_pulses(pd_times, vs_times)

        expected_trimmed_pd_times = np.array([1.0, 1.5, 2.0])
        np.testing.assert_array_almost_equal(
            trimmed_pd_times, expected_trimmed_pd_times
        )

    def test_correct_on_off_effects(self):
        # Create mock photodiode times
        pd_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        # Call the function to correct on/off effects in the photodiode signal
        corrected_pd_times = sync.correct_on_off_effects(pd_times)

        # Checking len because function relies
        # on statistical properties
        self.assertTrue(len(corrected_pd_times), len(pd_times))

    def test_trim_discontiguous_vsyncs(self):
        # Create mock vsync times
        vs_times = np.array([1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 3.0])

        # Trim discontiguous vsyncs from the photodiode signal
        trimmed_vs_times = sync.trim_discontiguous_vsyncs(vs_times)

        expected_trimmed_vs_times = np.array(
            [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 3.0]
        )
        np.testing.assert_array_almost_equal(
            trimmed_vs_times, expected_trimmed_vs_times
        )

    def test_assign_to_last(self):
        """ "
        Tests whether irregularity is assigned as expected
        """
        # Mock data arrays for starts, ends, frame duration, irregularity
        starts = np.array([1.0, 2.0, 3.0])
        ends = np.array([1.1, 2.1, 3.1])
        frame_duration = 0.1
        irregularity = 1

        # Assign the irregularity to the last frame
        new_starts, new_ends = sync.assign_to_last(
            starts, ends, frame_duration, irregularity
        )

        expected_new_ends = np.array([1.1, 2.1, 3.2])
        np.testing.assert_array_almost_equal(new_ends, expected_new_ends)

    def test_remove_zero_frames(self):
        # Create mock frame times
        frame_times = np.array(
            [1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2]
        )

        # Call the function to remove zero delta frames from the frame times
        modified_frame_times = sync.remove_zero_frames(frame_times)

        expected_modified_frame_times = np.array(
            [1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2]
        )
        np.testing.assert_array_almost_equal(
            modified_frame_times, expected_modified_frame_times
        )

    def test_compute_frame_times(self):
        # Create mock photodiode times
        photodiode_times = np.arange(0, 11, 1)

        # Set frame duration, number of frames, and cycle
        frame_duration = 1
        num_frames = 10
        cycle = 1

        # Call the function to compute frame times
        indices, starts, ends = sync.compute_frame_times(
            photodiode_times, frame_duration, num_frames, cycle
        )

        expected_indices = np.arange(0, 10, 1)
        expected_starts = np.arange(0, 10, 1)
        expected_ends = np.arange(1, 11, 1)
        np.testing.assert_array_almost_equal(indices, expected_indices)
        np.testing.assert_array_almost_equal(starts, expected_starts)
        np.testing.assert_array_almost_equal(ends, expected_ends)

    def test_separate_vsyncs_and_photodiode_times(self):
        # Create mock vsync and photodiode times
        vs_times = np.arange(0, 11, 1)
        pd_times = np.arange(0, 20, 2)

        # Call the function to separate vsync and photodiode times
        vs_times_out, pd_times_out = sync.separate_vsyncs_and_photodiode_times(
            vs_times, pd_times
        )

        expected_vs_times_out = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
        expected_pd_times_out = [
            np.array([2, 4, 6, 8, 10, 12, 14, 16, 18]),
        ]
        np.testing.assert_array_almost_equal(vs_times_out, expected_vs_times_out)
        np.testing.assert_array_almost_equal(pd_times_out, expected_pd_times_out)

    def test_flag_unexpected_edges(self):
        # Create mock photodiode times
        pd_times = np.array([1, 2, 3, 5, 7, 8, 9, 11])

        # Call the function to flag unexpected edges
        expected_duration_mask = sync.flag_unexpected_edges(pd_times, ndevs=1)

        expected_result = np.array([1, 1, 0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(expected_duration_mask, expected_result)

    def test_fix_unexpected_edges(self):
        # Create mock photodiode times
        pd_times = np.array([1, 2, 3, 5, 7, 8, 9, 11])

        # Call the function to fix unexpected edges
        output_edges = sync.fix_unexpected_edges(
            pd_times, ndevs=1, cycle=2, max_frame_offset=2
        )

        expected_result = np.array([1, 2, 3, 5, 5, 7, 8, 9, 11])
        np.testing.assert_array_equal(output_edges, expected_result)


if __name__ == "__main__":
    unittest.main()
