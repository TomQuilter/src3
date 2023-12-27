import unittest
import pandas as pd
import torch

from utils.metric_utils.calc_metric import calc_acc, calc_conf_matrix

class TestCalcMetric(unittest.TestCase):
    def test_calc_acc(self):
        # Testcase
        data_ts = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        predictions = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
        acc = calc_acc(data_ts, predictions)

        self.assertEqual(acc, 80.0)

    
    def test_calc_conf_matrix(self):
        # Testcase
        data_ts = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        predictions = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
        conf_matrix = calc_conf_matrix(data_ts, predictions)

        # True
        T0, F1, F0, T1 = 1, 1, 0, 3
        true_conf_matrix = [[(T0/5)*100, (F1/5)*100], [(F0/5)*100, (T1/5)*100]]
        
        self.assertEqual(conf_matrix, true_conf_matrix)


if __name__ == '__main__':
    unittest.main()
