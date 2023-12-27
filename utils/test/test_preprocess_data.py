import unittest
import pandas as pd
import numpy as np

from utils.preprocess_data import preprocess_data

class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [4.0, 2.0, 1.0, 10.0],
            [nan, nan, nan, nan],
            [4.0, 0.0, 3.0, 0.0],
            [5.0, 2.0, 1.0, 8.0],
            [5.0, 2.0, 2.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=['q1', 'q2', 'q3', 'q4'])
        data_df.set_index(pd.Series([2,3,4,5,6]), inplace=True) # set row index

        meta_data = np.array([
            [5.0, 2.0, 3.0, 3.0],
            ['A', 'B', 'C', 'D'],
            [1, 1, 1, 3],
            [171, 52, 200, 183]
        ])
        meta_df = pd.DataFrame(data=meta_data, columns=['q1', 'q2', 'q3', 'q4'])
        meta_df.set_index(pd.Index(['Max', 'Topics', 'Difficulty', 'qtype']), inplace=True)
        
        processed_df, processed_meta_df = preprocess_data(data_df, meta_df)

        # True
        true_data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        true_df = pd.DataFrame(true_data, columns=[0, 1, 2, 3])

        true_meta_data = np.array([
            [5.0, 2.0, 3.0, 3.0],
            ['A', 'B', 'C', 'D'],
            [1, 1, 1, 3],
            [171, 52, 200, 183]
        ])
        true_meta_df = pd.DataFrame(data=true_meta_data, columns=[0, 1, 2, 3])
        true_meta_df.set_index(pd.Index(['Max', 'Topics', 'Difficulty', 'qtype']), inplace=True)

        self.assertTrue(processed_df.equals(true_df))
        self.assertTrue(processed_meta_df.equals(true_meta_df))


if __name__ == '__main__':
    unittest.main()
