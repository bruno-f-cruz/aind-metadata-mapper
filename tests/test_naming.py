import unittest

import pandas as pd
import numpy as np

from aind_metadata_mapper.utils import naming_utils as naming


class TestDropEmptyColumns(unittest.TestCase):
    def test_drop_empty_columns_all_nan(self):
        # Create a DataFrame with some columns all NaN
        data = {
            "A": [1, 2, 3],
            "B": [None, None, None],
            "C": [4, 5, 6],
            "D": [None, None, None],
        }
        df = pd.DataFrame(data)

        # Expected DataFrame after dropping columns B and D
        expected_data = {"A": [1, 2, 3], "C": [4, 5, 6]}
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.drop_empty_columns(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_empty_columns_no_nan(self):
        # Create a DataFrame with no columns all NaN
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = pd.DataFrame(data)

        # Expected DataFrame (unchanged)
        expected_df = df.copy()

        # Call the function and assert the result
        result_df = naming.drop_empty_columns(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_empty_columns_some_nan(self):
        # Create a DataFrame with some NaN values but not all in any column
        data = {"A": [1, None, 3], "B": [None, 2, 3], "C": [4, 5, 6]}
        df = pd.DataFrame(data)

        # Expected DataFrame (unchanged)
        expected_df = df.copy()

        # Call the function and assert the result
        result_df = naming.drop_empty_columns(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_empty_columns_all_empty(self):
        # Create a DataFrame with all columns containing only NaN values
        data = {
            "A": [None, None, None],
            "B": [None, None, None],
            "C": [None, None, None],
        }
        df = pd.DataFrame(data)

        # Expected DataFrame (empty DataFrame)
        expected_df = pd.DataFrame()

        # Call the function and assert the result
        result_df = naming.drop_empty_columns(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_collapse_columns_merge(self):
        # Create a DataFrame with columns that can be merged
        data = {
            "A": [1, None, None],
            "b": [None, 2, None],
            "C": [None, None, 3],
        }
        df = pd.DataFrame(data)

        # Expected DataFrame after merging columns
        expected_data = {"A": [1, 2, 3]}
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.collapse_columns(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_collapse_columns_no_merge(self):
        # Create a DataFrame with columns that cannot be merged
        data = {
            "A": [1, None, None],
            "B": [None, 2, None],
            "C": [None, None, 3],
        }
        df = pd.DataFrame(data)

        # Expected DataFrame (unchanged)
        expected_df = df.copy()

        # Call the function and assert the result
        result_df = naming.collapse_columns(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_collapse_columns_merge_with_overwrite(self):
        # Create a DataFrame with columns that can be merged, with some overlapping non-NaN values
        data = {
            "A": [1, None, None],
            "B": [None, 2, None],
            "C": [None, 3, None],
            "a": [None, 4, None],
            "b": [5, None, None],
            "c": [None, None, 6],
        }
        df = pd.DataFrame(data)

        # Expected DataFrame after merging columns with overwritten NaN values
        expected_data = {
            "A": [1, 4, None],
            "B": [5, 2, None],
            "C": [None, 3, 6],
        }
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.collapse_columns(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_add_number_to_shuffled_movie_no_matching_rows(self):
        # Create a DataFrame with no rows matching the shuffled movie regex
        data = {
            "stim_name": [
                "natural_movie_1",
                "natural_movie_2",
                "natural_movie_3",
            ]
        }
        df = pd.DataFrame(data)

        # Expected DataFrame (unchanged)
        expected_df = df.copy()

        # Call the function and assert the result
        result_df = naming.add_number_to_shuffled_movie(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_add_number_to_shuffled_movie_multiple_movie_numbers(self):
        # Create a DataFrame with multiple different movie numbers
        data = {
            "stim_name": [
                "natural_movie_1_shuffled",
                "natural_movie_2_shuffled",
                "natural_movie_3_shuffled",
            ]
        }
        df = pd.DataFrame(data)

        # Call the function and assert that it raises a ValueError
        with self.assertRaises(ValueError):
            naming.add_number_to_shuffled_movie(df)

    def test_add_number_to_shuffled_movie_single_movie_number(self):
        # Create a DataFrame with a single movie number
        data = {
            "stim_name": [
                "natural_movie_1_shuffled",
                "natural_movie_1_shuffled",
                "natural_movie_1_shuffled",
            ]
        }
        df = pd.DataFrame(data)

        # Expected DataFrame with the stim_name column modified
        expected_data = {
            "stim_name": [
                "natural_movie_1",
                "natural_movie_1",
                "natural_movie_1",
            ]
        }
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.add_number_to_shuffled_movie(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_add_number_to_shuffled_movie_mixed_columns(self):
        # Create a DataFrame with mixed columns including rows matching the shuffled movie regex
        data = {
            "stim_name": [
                "natural_movie_1_shuffled",
                "image1.jpg",
                "natural_movie_2_shuffled",
                "natural_movie_3_shuffled",
            ]
        }
        df = pd.DataFrame(data)

        # Expected DataFrame with only the matching rows modified
        expected_data = {
            "stim_name": [
                "natural_movie_1",
                "image1.jpg",
                "natural_movie_2",
                "natural_movie_3",
            ]
        }
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.add_number_to_shuffled_movie(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_stimulus_names_no_mapping(self):
        # Create a DataFrame with no mapping provided
        data = {"stim_name": ["stim1", "stim2", "stim3"]}
        df = pd.DataFrame(data)

        # Expected DataFrame (unchanged)
        expected_df = df.copy()

        # Call the function and assert the result
        result_df = naming.map_stimulus_names(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_stimulus_names_with_mapping(self):
        # Create a DataFrame with a mapping provided
        data = {"stim_name": ["stim1", "stim2", "stim3"]}
        df = pd.DataFrame(data)
        name_map = {"stim1": "new_stim1", "stim3": "new_stim3"}

        # Expected DataFrame with stim_name column modified according to the mapping
        expected_data = {"stim_name": ["new_stim1", "stim2", "new_stim3"]}
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.map_stimulus_names(df, name_map=name_map)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_stimulus_names_with_nan_mapping(self):
        # Create a DataFrame with a mapping provided including NaN
        data = {"stim_name": ["stim1", "stim2", np.nan]}
        df = pd.DataFrame(data)
        name_map = {"stim1": "new_stim1", np.nan: "new_spontaneous"}

        # Expected DataFrame with stim_name column modified according to the mapping
        expected_data = {
            "stim_name": ["new_stim1", "stim2", "new_spontaneous"]
        }
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.map_stimulus_names(df, name_map=name_map)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_stimulus_names_with_column_name(self):
        # Create a DataFrame with a custom stimulus column name
        data = {"custom_stimulus_name": ["stim1", "stim2", "stim3"]}
        df = pd.DataFrame(data)
        name_map = {"stim1": "new_stim1", "stim3": "new_stim3"}

        # Expected DataFrame with custom_stimulus_name column modified according to the mapping
        expected_data = {
            "custom_stimulus_name": ["new_stim1", "stim2", "new_stim3"]
        }
        expected_df = pd.DataFrame(expected_data)

        # Call the function with the custom column name and assert the result
        result_df = naming.map_stimulus_names(
            df, name_map=name_map, stim_colname="custom_stimulus_name"
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_column_names_no_mapping(self):
        # Create a DataFrame with no mapping provided
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = pd.DataFrame(data)

        # Expected DataFrame (unchanged)
        expected_df = df.copy()

        # Call the function and assert the result
        result_df = naming.map_column_names(df)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_column_names_with_mapping(self):
        # Create a DataFrame with a mapping provided
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = pd.DataFrame(data)
        name_map = {"A": "X", "B": "Y", "C": "Z"}

        # Expected DataFrame with column names modified according to the mapping
        expected_data = {"X": [1, 2, 3], "Y": [4, 5, 6], "Z": [7, 8, 9]}
        expected_df = pd.DataFrame(expected_data)

        # Call the function and assert the result
        result_df = naming.map_column_names(df, name_map=name_map)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_column_names_with_ignore_case(self):
        # Create a DataFrame with a mapping provided and ignore_case=True
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = pd.DataFrame(data)
        name_map = {"a": "X", "b": "Y", "C": "Z"}

        # Expected DataFrame with column names modified according to the mapping, ignoring case
        expected_data = {"X": [1, 2, 3], "Y": [4, 5, 6], "Z": [7, 8, 9]}
        expected_df = pd.DataFrame(expected_data)

        # Call the function with ignore_case=True and assert the result
        result_df = naming.map_column_names(
            df, name_map=name_map, ignore_case=True
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_map_column_names_with_ignore_case_false(self):
        # Create a DataFrame with a mapping provided and ignore_case=False
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = pd.DataFrame(data)
        name_map = {"a": "X", "b": "Y", "C": "Z"}

        # Expected DataFrame (unchanged) because ignore_case=False and column names are case-sensitive
        expected_df = df.copy()

        # Call the function with ignore_case=False and assert the result
        result_df = naming.map_column_names(
            df, name_map=name_map, ignore_case=False
        )
        pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == "__main__":
    unittest.main()
