import unittest
import torch

from utils.ts_utils.shuffle_ts import shuffle_1D_vect_ts, shuffle_2D_vect_ts


class TestShuffleTs(unittest.TestCase):
    def test_shuffle_1D_vect_ts(self):
        # Testcase: seed=1001
        vect_ts = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
        shuffled_ts = shuffle_1D_vect_ts(vect_ts, shuffle_seed=1001)

        # True
        true_ts = torch.tensor([1, 0, 0, 1], dtype=torch.int32)

        self.assertTrue(torch.equal(shuffled_ts, true_ts))


    def test_shuffle_2D_vect_ts(self):
        # Testcase: seed=1001
        vect_ts = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ], dtype=torch.int32)

        shuffled_ts = shuffle_2D_vect_ts(vect_ts, shuffle_seed=1001)

        # True
        true_ts = torch.tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0]
        ], dtype=torch.int32)

        self.assertTrue(torch.equal(shuffled_ts, true_ts))


if __name__ == '__main__':
    unittest.main()
