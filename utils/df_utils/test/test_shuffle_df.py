import unittest
import pandas as pd

from utils.df_utils.shuffle_df import shuffle_rows, shuffle_cols

class TestShuffleDf(unittest.TestCase):
    def test_shuffle_rows_1(self):
        nan = float('nan')
        
        # Testcase: seed=1001, reset_index=False
        data = [
            [1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, nan],
            [0.0, 0.0, 0.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3]) # set col index
        data_df.set_index(pd.Series([6, 7, 8]), inplace=True) # set row index

        shuffled_df = shuffle_rows(data_df, shuffle_seed=1001, reset_index=False)

        # True
        true_data = [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [nan, nan, nan, nan]
        ]
        true_df = pd.DataFrame(true_data, columns=[0, 1, 2, 3]) # set col index
        true_df.set_index(pd.Series([6, 8, 7]), inplace=True) # set row index

        self.assertTrue(shuffled_df.equals(true_df))


    def test_shuffle_rows_2(self):
        nan = float('nan')
        
        # Testcase: seed=1001, reset_index=True
        data = [
            [1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, nan],
            [0.0, 0.0, 0.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3]) # set col index
        data_df.set_index(pd.Series([6, 7, 8]), inplace=True) # set row index

        shuffled_df = shuffle_rows(data_df, shuffle_seed=1001, reset_index=True)

        # True
        true_data = [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [nan, nan, nan, nan]
        ]
        true_df = pd.DataFrame(true_data, columns=[0, 1, 2, 3])

        self.assertTrue(shuffled_df.equals(true_df))


    def test_shuffle_cols_1(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [1.0, 1.0, 1.0, nan],
            [nan, nan, nan, nan],
            [1.0, 1.0, 1.0, nan]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3])
        shuffled_df = shuffle_cols(data_df, shuffle_seed=1001, reset_index=False)

        # True
        true_data = [
            [1.0, 1.0, nan, 1.0],
            [nan, nan, nan, nan],
            [1.0, 1.0, nan, 1.0]
        ]
        true_df = pd.DataFrame(true_data, columns=[0, 2, 3, 1])

        self.assertTrue(shuffled_df.equals(true_df))


    def test_shuffle_cols_2(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [1.0, 1.0, 1.0, nan],
            [nan, nan, nan, nan],
            [1.0, 1.0, 1.0, nan]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3])
        shuffled_df = shuffle_cols(data_df, shuffle_seed=1001, reset_index=True)

        # True
        true_data = [
            [1.0, 1.0, nan, 1.0],
            [nan, nan, nan, nan],
            [1.0, 1.0, nan, 1.0]
        ]
        true_df = pd.DataFrame(true_data, columns=[0, 1, 2, 3])

        self.assertTrue(shuffled_df.equals(true_df))


if __name__ == '__main__':
    unittest.main()
