import unittest
import pandas as pd
import torch


from utils.df_utils.tensorise_df import tensorise_df

class TestTensoriseDf(unittest.TestCase):
    def test_tensorise_df(self):
        # Testcase: float
        data = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3]) # set col index
        data_ts = tensorise_df(data_df)


        # True
        true_ts = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.float64)
        
        self.assertTrue(torch.equal(data_ts, true_ts))


    def test_tensorise_df_2(self):
        # Testcase: int
        data = [
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3]) # set col index
        data_ts = tensorise_df(data_df)


        # True
        true_ts = torch.tensor([
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 0]
        ])
        
        self.assertTrue(torch.equal(data_ts, true_ts))


if __name__ == '__main__':
    unittest.main()
