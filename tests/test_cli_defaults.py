import sys
import unittest
from unittest import mock

import miner


class CliDefaultTests(unittest.TestCase):
    def run_cli(self, has_sme):
        run_pool = mock.Mock()
        argv = [
            "miner.py",
            "--pool", "localhost:3360",
            "--address", "prl1ptest",
            "--max-draws", "1",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(miner, "_has_sme", return_value=has_sme),
            mock.patch.object(miner, "run_pool", run_pool),
            mock.patch.object(miner, "_make_logger", return_value=mock.Mock()),
        ):
            miner.main()
        return run_pool

    def test_m4_defaults_to_peak_both_mode_configuration(self):
        run_pool = self.run_cli(has_sme=True)
        args, kwargs = run_pool.call_args
        self.assertEqual(args[3:6], (16384, 2048, 128))
        self.assertEqual(kwargs["engine"], "both")
        self.assertEqual(kwargs["m_cpu"], 8192)
        self.assertEqual(kwargs["compute_pct"], 100)

    def test_pre_m4_defaults_to_gpu_fallback(self):
        run_pool = self.run_cli(has_sme=False)
        self.assertEqual(run_pool.call_args.kwargs["engine"], "gpu")

    def test_pool_password_is_forwarded(self):
        run_pool = mock.Mock()
        argv = [
            "miner.py", "--pool", "us1.alphapool.tech:5571",
            "--address", "prl1ptest", "--difficulty", "20000",
            "--password", "x;d=20000;t=25", "--max-draws", "1",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(miner, "_has_sme", return_value=True),
            mock.patch.object(miner, "run_pool", run_pool),
            mock.patch.object(miner, "_make_logger", return_value=mock.Mock()),
        ):
            miner.main()
        self.assertEqual(run_pool.call_args.kwargs["difficulty"], 20000)
        self.assertEqual(run_pool.call_args.kwargs["password"], "x;d=20000;t=25")


if __name__ == "__main__":
    unittest.main()
