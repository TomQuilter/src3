import unittest
import pandas as pd
import numpy as np

from utils.split_train_test import split_test_questions

class TestSplitTrainTest(unittest.TestCase):
    def test_split_train_test(self):
        split_params = {
            'seed': 1000, 
            'random_q_order': False, 
            'random_s_order': True, 
            'test_q_params': {
                'q_range': [0, 70], 
                'specific': None
                }, 
            'test_split': 20, 
            'val_split': 10
            }
        #TODO


    def test_split_test_questions_1(self):
        nan = float('nan')

        # Testcase: range of questions involed in test set
        data = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3, 4, 5])
        selected_df, remaining_df = split_test_questions(data_df, q_range=[0, 3], specific=None)

        true_selected = [
            [1.0, 1.0, 1.0],
            [nan, nan, nan],
            [1.0, 0.0, 1.0]
        ]
        true_selected_df = pd.DataFrame(true_selected, columns=[0, 1, 2])
        true_remaining = [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
        true_remaining_df = pd.DataFrame(true_remaining, columns=[3, 4, 5])

        self.assertTrue(selected_df.equals(true_selected_df))
        self.assertTrue(remaining_df.equals(true_remaining_df))


    def test_split_test_questions_2(self):
        nan = float('nan')

        # Testcase: range of questions (all questions) involed in test set
        data = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3, 4, 5])
        selected_df, remaining_df = split_test_questions(data_df, q_range=[0, 6], specific=None)

        true_selected = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ]
        true_selected_df = pd.DataFrame(true_selected, columns=[0, 1, 2, 3, 4, 5])
        true_remaining_df = pd.DataFrame(index=pd.Index([0,1,2]))

        self.assertTrue(selected_df.equals(true_selected_df))
        self.assertTrue(remaining_df.equals(true_remaining_df))


    def test_split_test_questions_3(self):
        nan = float('nan')

        # Testcase: specific questions involved in test set
        data = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3, 4, 5])
        selected_df, remaining_df = split_test_questions(data_df, q_range=None, specific=[1, 3, 5])

        true_selected = [
            [1.0, 1.0, 1.0],
            [nan, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]
        true_selected_df = pd.DataFrame(true_selected, columns=[1, 3, 5])
        true_remaining = [
            [1.0, 1.0, 1.0],
            [nan, nan, 0.0],
            [1.0, 1.0, 1.0]
        ]
        true_remaining_df = pd.DataFrame(true_remaining, columns=[0, 2, 4])

        self.assertTrue(selected_df.equals(true_selected_df))
        self.assertTrue(remaining_df.equals(true_remaining_df))

    
    def test_split_test_questions_4(self):
        nan = float('nan')

        # Testcase: all questions involed in test set
        data = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=[0, 1, 2, 3, 4, 5])
        selected_df, remaining_df = split_test_questions(data_df, q_range=None, specific=None)

        true_selected = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [nan, nan, nan, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        ]
        true_selected_df = pd.DataFrame(true_selected, columns=[0, 1, 2, 3, 4, 5])
        true_remaining_df = pd.DataFrame(index=pd.Index([0,1,2]))

        self.assertTrue(selected_df.equals(true_selected_df))
        self.assertTrue(remaining_df.equals(true_remaining_df))


if __name__ == '__main__':
    unittest.main()
