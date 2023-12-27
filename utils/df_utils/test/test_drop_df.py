import unittest
import pandas as pd

from utils.df_utils.drop_df import drop_nan_rows


class TestDropDf(unittest.TestCase):
    def test_drop_nan_rows(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[1, 2, 3, 4])
        data_df.set_index(pd.Series([2,3,4,5,6]), inplace=True) # set row index
        data_df = drop_nan_rows(data_df)

        # True
        true_data = [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        true_df = pd.DataFrame(true_data, columns=[1, 2, 3, 4])
        true_df.set_index(pd.Series([2,4,5,6]), inplace=True) # set row index
        print(data_df)
        print(true_df)

        self.assertTrue(data_df.equals(true_df))


if __name__ == '__main__':
    unittest.main()
