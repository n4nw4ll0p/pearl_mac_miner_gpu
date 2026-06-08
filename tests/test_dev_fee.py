import math
import unittest

import dev_fee


class DevFeeSchedulerTests(unittest.TestCase):
    def test_default_fee_reaches_one_percent(self):
        scheduler = dev_fee.DevFeeScheduler(min_slice_s=1.0)
        scheduler.record_user(99.0)

        self.assertAlmostEqual(scheduler.seconds_owed(), 1.0)
        scheduler.record_dev(1.0)
        self.assertAlmostEqual(scheduler.effective_pct, 1.0)
        self.assertEqual(scheduler.seconds_owed(), 0.0)

    def test_default_minimum_slice_is_due_after_99_minutes(self):
        scheduler = dev_fee.DevFeeScheduler()
        scheduler.record_user(5_939.0)
        self.assertEqual(scheduler.seconds_owed(), 0.0)

        scheduler.record_user(2.0)
        self.assertGreaterEqual(scheduler.seconds_owed(), 60.0)

    def test_fee_converges_after_repeated_slices(self):
        scheduler = dev_fee.DevFeeScheduler(min_slice_s=1.0)
        for _ in range(10):
            scheduler.record_user(99.0)
            scheduler.record_dev(scheduler.seconds_owed())

        self.assertAlmostEqual(scheduler.effective_pct, 1.0)

    def test_zero_fee_is_a_permanent_noop(self):
        scheduler = dev_fee.DevFeeScheduler(percent=0.0)
        scheduler.record_user(1_000_000.0)
        self.assertEqual(scheduler.seconds_owed(), 0.0)
        self.assertEqual(scheduler.effective_pct, 0.0)

    def test_negative_durations_are_ignored(self):
        scheduler = dev_fee.DevFeeScheduler(min_slice_s=1.0)
        scheduler.record_user(-10.0)
        scheduler.record_dev(-10.0)
        self.assertEqual(scheduler.user_s, 0.0)
        self.assertEqual(scheduler.dev_s, 0.0)

    def test_invalid_configuration_is_rejected(self):
        for percent in (-0.1, 100.0, math.inf, math.nan):
            with self.subTest(percent=percent):
                with self.assertRaises(ValueError):
                    dev_fee.DevFeeScheduler(percent=percent)
        for seconds in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(seconds=seconds):
                with self.assertRaises(ValueError):
                    dev_fee.DevFeeScheduler(min_slice_s=seconds)


if __name__ == "__main__":
    unittest.main()
