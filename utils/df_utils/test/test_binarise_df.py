import unittest
import pandas as pd

from utils.df_utils.binarise_df import binarise_df

class TestBinariseDf(unittest.TestCase):
    def test_binarise_df_1(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [4.0, 2.0, 1.0, 3.0],
            [nan, nan, nan, nan],
            [4.0, 0.0, 3.0, 0.0],
            [5.0, 2.0, 1.0, 3.0],
            [5.0, 2.0, 2.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=['q1', 'q2', 'q3', 'q4'])
        
        max_scores_data = {'q1':5.0, 'q2':2.0, 'q3':3.0, 'q4':3.0}
        max_scores_df = pd.Series(data=max_scores_data, index=['q1', 'q2', 'q3', 'q4'])
        
        bin_df = binarise_df(data_df, max_scores_df)

        # True
        true_data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        true_df = pd.DataFrame(true_data, columns=['q1', 'q2', 'q3', 'q4'])

        self.assertTrue(bin_df.equals(true_df))


if __name__ == '__main__':
    unittest.main()
