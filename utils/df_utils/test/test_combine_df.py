import unittest
import pandas as pd

from utils.df_utils.combine_df import combine_df


class TestCombineDf(unittest.TestCase):
    def test_combine_df_1(self):
        nan = float('nan')

        # Testcase: retains original row index (i.e. student identity) after concatenation
        data1 = [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        data_df1 = pd.DataFrame(data1, columns=['q1', 'q2', 'q3', 'q4'])

        data2 = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0]
        ]
        data_df2 = pd.DataFrame(data2, columns=['q1.1', 'q2.1', 'q3.1', 'q4.1'])
        data_df2.set_index(pd.Series([2,3,4,5]), inplace=True) # set row index

        combined_df = combine_df([data_df1, data_df2])

        # True
        true_combined = [
            [1.0, 1.0, 0.0, 1.0, nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0, nan, nan, nan, nan], 
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, nan, nan, nan, nan],
            [nan, nan, nan, nan, 1.0, 0.0, 1.0, 0.0],
            [nan, nan, nan, nan, 1.0, 1.0, 0.0, 1.0]
        ]
        true_combined_df = pd.DataFrame(true_combined, columns=['q1', 'q2', 'q3', 'q4', 'q1.1', 'q2.1', 'q3.1', 'q4.1'])

        self.assertTrue(combined_df.equals(true_combined_df))


    def test_combine_df_2(self):
        nan = float('nan')

        # Testcase: rows with all entries == nan remain
        data1 = [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan]
        ]
        data_df1 = pd.DataFrame(data1, columns=['q1', 'q2', 'q3', 'q4'])

        data2 = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0]
        ]
        data_df2 = pd.DataFrame(data2, columns=['q1.1', 'q2.1', 'q3.1', 'q4.1'])
        data_df2.set_index(pd.Series([2,3,4,5]), inplace=True) # set row index

        combined_df = combine_df([data_df1, data_df2])

        # True
        true_combined = [
            [1.0, 1.0, 0.0, 1.0, nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0, nan, nan, nan, nan], 
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, 1.0, 0.0, 1.0, 0.0],
            [nan, nan, nan, nan, 1.0, 1.0, 0.0, 1.0]
        ]
        true_combined_df = pd.DataFrame(true_combined, columns=['q1', 'q2', 'q3', 'q4', 'q1.1', 'q2.1', 'q3.1', 'q4.1'])

        self.assertTrue(combined_df.equals(true_combined_df))


    def test_combine_df_3(self):
        nan = float('nan')

        # Testcase: concat 3 df's
        data1 = [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan]
        ]
        data_df1 = pd.DataFrame(data1, columns=['q1', 'q2', 'q3', 'q4'])

        data2 = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0]
        ]
        data_df2 = pd.DataFrame(data2, columns=['q1.1', 'q2.1', 'q3.1', 'q4.1'])
        data_df2.set_index(pd.Series([2,3,4,5]), inplace=True) # set row index

        data3 = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [nan, nan, nan, nan]
        ]
        data_df3 = pd.DataFrame(data3, columns=['q1.2', 'q2.2', 'q3.2', 'q4.2'])

        combined_df = combine_df([data_df1, data_df2, data_df3])

        # True
        true_data = [
            [1.0, 1.0, 0.0, 1.0, nan, nan, nan, nan, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan], 
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, 1.0, 0.0, 1.0, 0.0, nan, nan, nan, nan],
            [nan, nan, nan, nan, 1.0, 1.0, 0.0, 1.0, nan, nan, nan, nan]
        ]
        true_df = pd.DataFrame(true_data, columns=['q1', 'q2', 'q3', 'q4', 'q1.1', 'q2.1', 'q3.1', 'q4.1', 'q1.2', 'q2.2', 'q3.2', 'q4.2'])

        self.assertTrue(combined_df.equals(true_df))


if __name__ == '__main__':
    unittest.main()
