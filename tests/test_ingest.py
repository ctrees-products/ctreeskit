import unittest

import numpy as np

from ctreeskit.arraylake_tools.ingest import AnnualRasterIngester


class TestIndexSlice(unittest.TestCase):
    """_index_slice must locate aligned windows and reject off-grid ones
    instead of silently snapping to the nearest index (issue #16)."""

    def setUp(self):
        self.stored = np.arange(0.0, 100.0, 1.0)

    def test_aligned_subset_round_trips(self):
        subset = self.stored[10:20]
        s = AnnualRasterIngester._index_slice(self.stored, subset)
        self.assertEqual(s, slice(10, 20))
        np.testing.assert_array_equal(self.stored[s], subset)

    def test_float_drift_within_tolerance_is_accepted(self):
        subset = self.stored[10:20] + 1e-9
        s = AnnualRasterIngester._index_slice(self.stored, subset)
        self.assertEqual(s, slice(10, 20))

    def test_off_grid_subset_raises(self):
        subset = self.stored[10:20] + 0.4  # off-origin by 0.4 of a cell
        with self.assertRaises(ValueError) as ctx:
            AnnualRasterIngester._index_slice(self.stored, subset)
        self.assertIn("align", str(ctx.exception))

    def test_different_resolution_raises(self):
        subset = np.arange(10.0, 20.0, 0.5)
        with self.assertRaises(ValueError):
            AnnualRasterIngester._index_slice(self.stored, subset)

    def test_subset_extending_beyond_grid_raises(self):
        subset = np.arange(95.0, 105.0, 1.0)
        with self.assertRaises(ValueError):
            AnnualRasterIngester._index_slice(self.stored, subset)

    def test_descending_coords_aligned(self):
        stored_desc = self.stored[::-1].copy()
        subset = stored_desc[10:20]
        s = AnnualRasterIngester._index_slice(stored_desc, subset)
        self.assertEqual(s, slice(10, 20))


if __name__ == "__main__":
    unittest.main()
