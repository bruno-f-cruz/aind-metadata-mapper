""" Unit tests for the stim_utils module in the utils package. """

import unittest

import pandas as pd
import numpy as np

from unittest.mock import MagicMock, patch
from aind_metadata_mapper.open_ephys.utils import stim_utils as stim


class TestStimUtils(unittest.TestCase):
    """
    Tests Stim utils
    """
    def test_convert_filepath_caseinsensitive(self):
        """
        Test the convert_filepath_caseinsensitive function.
        """
        # Test when "TRAINING" is in the filename
        self.assertEqual(stim.convert_filepath_caseinsensitive("some/TRAINING/file.txt"), "some/training/file.txt")

        # Test when "TRAINING" is not in the filename
        self.assertEqual(stim.convert_filepath_caseinsensitive("some/OTHER/file.txt"), "some/OTHER/file.txt")

        # Test when "TRAINING" is in the middle of the filename
        self.assertEqual(stim.convert_filepath_caseinsensitive("some/TRAINING/file/TRAINING.txt"), "some/training/file/training.txt")

        # Test when "TRAINING" is at the end of the filename
        self.assertEqual(stim.convert_filepath_caseinsensitive("some/file/TRAINING"), "some/file/training")

        # Test when filename is empty
        self.assertEqual(stim.convert_filepath_caseinsensitive(""), "")

        # Test when filename is just "TRAINING"
        self.assertEqual(stim.convert_filepath_caseinsensitive("TRAINING"), "training")


    def test_enforce_df_int_typing(self):
        """
        Test the enforce_df_int_typing function.
        """
        INT_NULL = -999  # Assuming this is the value set in the actual module

        # Create a sample DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, None],
            'B': [4, None, 6, 7],
            'C': ['foo', 'bar', 'baz', 'qux']
        })

        # Expected DataFrame without using pandas Int64 type
        expected_df_no_pandas_type = pd.DataFrame({
            'A': [1, 2, 3, INT_NULL],
            'B': [4, INT_NULL, 6, 7],
            'C': ['foo', 'bar', 'baz', 'qux']
        })

        # Expected DataFrame using pandas Int64 type
        expected_df_pandas_type = pd.DataFrame({
            'A': [1, 2, 3, pd.NA],
            'B': [4, pd.NA, 6, 7],
            'C': ['foo', 'bar', 'baz', 'qux']
        }, dtype={"A": "Int64", "B": "Int64"})

        # Test without using pandas Int64 type
        result_df_no_pandas_type = stim.enforce_df_int_typing(df.copy(), ['A', 'B'], use_pandas_type=False)
        pd.testing.assert_frame_equal(result_df_no_pandas_type, expected_df_no_pandas_type)

        # Test using pandas Int64 type
        result_df_pandas_type = stim.enforce_df_int_typing(df.copy(), ['A', 'B'], use_pandas_type=True)
        pd.testing.assert_frame_equal(result_df_pandas_type, expected_df_pandas_type)

        # Test with columns that are not in the DataFrame
        result_df_no_columns = stim.enforce_df_int_typing(df.copy(), ['D', 'E'], use_pandas_type=False)
        pd.testing.assert_frame_equal(result_df_no_columns, df)

        # Test with an empty DataFrame
        empty_df = pd.DataFrame()
        result_empty_df = stim.enforce_df_int_typing(empty_df, ['A', 'B'], use_pandas_type=False)
        pd.testing.assert_frame_equal(result_empty_df, empty_df)


    def test_enforce_df_column_order():
        """
        Test the enforce_df_column_order function.
        """
        # Create a sample DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
            'D': [10, 11, 12]
        })

        # Test case: Specified column order
        column_order = ['D', 'B']
        expected_df = pd.DataFrame({
            'D': [10, 11, 12],
            'B': [4, 5, 6],
            'A': [1, 2, 3],
            'C': [7, 8, 9]
        })
        result_df = stim.enforce_df_column_order(df, column_order)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # Test case: Specified column order with non-existing columns
        column_order = ['D', 'E', 'B']
        expected_df = pd.DataFrame({
            'D': [10, 11, 12],
            'B': [4, 5, 6],
            'A': [1, 2, 3],
            'C': [7, 8, 9]
        })
        result_df = stim.enforce_df_column_order(df, column_order)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # Test case: No specified column order
        column_order = []
        expected_df = df.copy()
        result_df = stim.enforce_df_column_order(df, column_order)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # Test case: Specified column order with all columns
        column_order = ['C', 'A', 'D', 'B']
        expected_df = pd.DataFrame({
            'C': [7, 8, 9],
            'A': [1, 2, 3],
            'D': [10, 11, 12],
            'B': [4, 5, 6]
        })
        result_df = stim.enforce_df_column_order(df, column_order)
        pd.testing.assert_frame_equal(result_df, expected_df)

        # Test case: Empty DataFrame
        empty_df = pd.DataFrame()
        column_order = ['A', 'B']
        result_df = stim.enforce_df_column_order(empty_df, column_order)
        pd.testing.assert_frame_equal(result_df, empty_df)


    def test_seconds_to_frames():
        """
        Test the seconds_to_frames function.
        """

        # Mock data
        seconds = [1.0, 2.5, 3.0]
        pkl_file = "test.pkl"
        pre_blank_sec = 0.5
        fps = 30

        # Expected result
        expected_frames = [45, 90, 105]

        # Mock pkl functions
        with patch("aind_metadata_mapper.open_ephys.utils.stim_utils.pkl.get_pre_blank_sec", return_value=pre_blank_sec):
            with patch("aind_metadata_mapper.open_ephys.utils.stim_utils.pkl.get_fps", return_value=fps):
                result_frames = stim.seconds_to_frames(seconds, pkl_file)
                np.testing.assert_array_equal(result_frames, expected_frames)

    def test_extract_const_params_from_stim_repr():
        """
        Test the extract_const_params_from_stim_repr function.
        """

        # Sample input data
        stim_repr = "param1=10, param2=[1, 2, 3], param3='value3', param4=4.5"

        # Mock patterns
        repr_params_re = re.compile(r'(\w+=[^,]+)')
        array_re = re.compile(r'^\[(?P<contents>.*)\]$')

        # Expected result
        expected_params = {
            'param1': 10,
            'param2': [1, 2, 3],
            'param3': 'value3',
            'param4': 4.5
        }

        # Mocking ast.literal_eval to correctly evaluate the string representations
        with patch("aind_metadata_mapper.open_ephys.utils.stim_utils.ast.literal_eval", side_effect=lambda x: eval(x)):
            result_params = stim.extract_const_params_from_stim_repr(stim_repr, repr_params_re, array_re)
            assert result_params == expected_params