import unittest
import pandas as pd
import torch

from utils.df_utils.vectorise_df import drop_nan_entries, vectorise_df


class TestVectoriseDf(unittest.TestCase):
    def test_vectorise_df_1(self):
        # Testcase: no nan entries
        data = [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3])
        vectorised_ts = vectorise_df(data_df)

        # True
        true_ts = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 2, 3, 0, 1, 2, 3]
        ], dtype=torch.int32)

        self.assertTrue(torch.equal(vectorised_ts, true_ts))


    def test_vectorise_df_2(self):
        nan = float('nan')
        
        # Testcase: contains nan
        data = [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, nan, nan]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3])
        vectorised_ts = vectorise_df(data_df)

        # True
        true_ts = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 1, 2, 3, 0, 1]
        ], dtype=torch.int32)

        self.assertTrue(torch.equal(vectorised_ts, true_ts))


    def test_drop_nan_entries(self):
        nan = float('nan')
        
        # Testcase
        data_ts = torch.tensor([
            [1, 1, 1, 1, 0, 0, nan, nan],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 2, 3, 0, 1, 2, 3]
        ], dtype=torch.float64)

        dropped_ts = drop_nan_entries(data_ts)

        # True
        true_ts = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 1, 2, 3, 0, 1]
        ], dtype=torch.float64)

        self.assertTrue(torch.equal(dropped_ts, true_ts))


if __name__ == '__main__':
    unittest.main()
