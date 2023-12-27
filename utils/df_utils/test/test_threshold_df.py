import unittest
import pandas as pd

from utils.df_utils.threshold_df import threshold_df


class TestThresholdDf(unittest.TestCase):
    def test_threshold_df_1(self):
        # Testcase
        data = [
            [4.0, 2.0, 1.0, 3.0],
            [4.0, 2.0, 0.0, 3.0],
            [4.0, 0.0, 3.0, 0.0],
            [5.0, 2.0, 1.0, 3.0],
            [6.0, 2.0, 2.0, -1.0]
        ]
        data_df = pd.DataFrame(data, columns=['q1', 'q2', 'q3', 'q4'])
        max_scores_data = {'q1':5.0, 'q2':2.0, 'q3':3.0, 'q4':3.0}
        max_scores_df = pd.Series(data=max_scores_data, index=['q1', 'q2', 'q3', 'q4'])
        thres_data_df = threshold_df(data_df, max_scores_df)

        # True
        true_thres_data = [
            [4.0, 2.0, 1.0, 3.0],
            [4.0, 2.0, 0.0, 3.0],
            [4.0, 0.0, 3.0, 0.0],
            [5.0, 2.0, 1.0, 3.0],
            [5.0, 2.0, 2.0, 0.0]
        ]
        true_thres_data_df = pd.DataFrame(true_thres_data, columns=['q1', 'q2', 'q3', 'q4'])

        self.assertTrue(thres_data_df.equals(true_thres_data_df))


    def test_threshold_df_2(self):
        nan = float('nan')

        # Testcase
        data = [
            [4.0, 2.0, 1.0, 3.0],
            [nan, nan, nan, nan],
            [4.0, 0.0, 3.0, 0.0],
            [5.0, 2.0, 1.0, 3.0],
            [6.0, 2.0, 2.0, -1.0]
        ]
        data_df = pd.DataFrame(data, columns=['q1', 'q2', 'q3', 'q4'])
        max_scores_data = {'q1':5.0, 'q2':2.0, 'q3':3.0, 'q4':3.0}
        max_scores_df = pd.Series(data=max_scores_data, index=['q1', 'q2', 'q3', 'q4'])
        thres_data_df = threshold_df(data_df, max_scores_df)

        # True
        true_thres_data = [
            [4.0, 2.0, 1.0, 3.0],
            [nan, nan, nan, nan],
            [4.0, 0.0, 3.0, 0.0],
            [5.0, 2.0, 1.0, 3.0],
            [5.0, 2.0, 2.0, 0.0]
        ]
        true_thres_data_df = pd.DataFrame(true_thres_data, columns=['q1', 'q2', 'q3', 'q4'])

        self.assertTrue(thres_data_df.equals(true_thres_data_df))


if __name__ == '__main__':
    unittest.main()
