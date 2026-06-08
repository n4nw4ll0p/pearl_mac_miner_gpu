import unittest
from unittest import mock

from bitcoinutils.bech32 import Encoding, bech32_decode, convertbits

import dev_fee
from pearl_core.pearld_rpc import PearldRpc, public_rpc_url


class ReleasePathTests(unittest.TestCase):
    def test_developer_address_is_a_prl_taproot_address(self):
        hrp, data, encoding = bech32_decode(dev_fee.DEV_FEE_ADDRESS)
        self.assertEqual(hrp, "prl")
        self.assertEqual(encoding, Encoding.BECH32M)
        self.assertEqual(data[0], 1)
        self.assertEqual(len(bytes(convertbits(data[1:], 5, 8, False))), 32)

    def test_rpc_credentials_and_query_are_redacted(self):
        url = "http://alice:secret@[::1]:18334/wallet/main?token=hidden#fragment"
        self.assertEqual(public_rpc_url(url), "http://[::1]:18334/wallet/main")
        self.assertEqual(PearldRpc(url).url, "http://[::1]:18334/wallet/main")

    def test_fee_slice_uses_disclosed_pool_and_address(self):
        logs = []

        def fake_mine_pool(*args, **kwargs):
            kwargs["search_time_callback"](60.25)
            return 60.25

        mine_pool = mock.Mock(side_effect=fake_mine_pool)
        mined = dev_fee.mine_slice(
            mine_pool, 60.0, 64, 2048, 128, engine="gpu", m_cpu=0,
            compute_pct=100, compute_pct_cpu=None, log=logs.append,
        )

        args, kwargs = mine_pool.call_args
        self.assertEqual(args[:3], (dev_fee.DEV_FEE_POOL, dev_fee.DEV_FEE_ADDRESS, "devfee"))
        self.assertEqual(kwargs["run_seconds"], 60.0)
        self.assertEqual(kwargs["stall_timeout"], 15.0)
        self.assertAlmostEqual(mined, 60.25)
        self.assertTrue(any("completed" in line for line in logs))

    def test_partial_fee_search_is_counted_when_pool_fails(self):
        logs = []

        def fake_mine_pool(*args, **kwargs):
            kwargs["search_time_callback"](7.5)
            raise RuntimeError("test disconnect")

        mined = dev_fee.mine_slice(
            fake_mine_pool, 60.0, 64, 2048, 128, engine="gpu", m_cpu=0,
            compute_pct=100, compute_pct_cpu=None, log=logs.append,
        )

        self.assertEqual(mined, 7.5)
        self.assertTrue(any("pool unavailable" in line for line in logs))

if __name__ == "__main__":
    unittest.main()
