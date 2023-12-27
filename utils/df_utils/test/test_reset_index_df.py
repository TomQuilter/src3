import unittest
import pandas as pd

from utils.df_utils.reset_index_df import reset_row_index, reset_col_index


class TestResetIndexDf(unittest.TestCase):
    def test_reset_row_index_1(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=['q1', 'q2', 'q3', 'q4'])
        data_df.set_index(pd.Series([6,7,8,9,10]), inplace=True) # set row index

        reset_df = reset_row_index(data_df)

        # True
        true_data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        true_df = pd.DataFrame(true_data, columns=['q1', 'q2', 'q3', 'q4'])

        self.assertTrue(reset_df.equals(true_df))


    def test_reset_col_index_1(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=['q1', 'q2', 'q3', 'q4'])
        reset_df = reset_col_index(data_df)

        # True
        true_data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        true_df = pd.DataFrame(true_data, columns=[0, 1, 2, 3])

        self.assertTrue(reset_df.equals(true_df))


if __name__ == '__main__':
    unittest.main()
