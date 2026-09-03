import unittest

from pearl_core.stratum import StratumPoolClient


class RecordingClient(StratumPoolClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, keepalive_interval=0, **kwargs)
        self.calls = []

    def _request(self, method, params, timeout=None):
        self.calls.append((method, params, timeout))
        return True


class StratumClientTests(unittest.TestCase):
    def test_authorize_uses_canonical_named_params(self):
        client = RecordingClient(
            "pool.example:1234", "prl1ptest", "mac", difficulty=20000,
            agent="pearl-apple-miner/1.0.3",
        )
        self.assertTrue(client.authorize())
        _, params, _ = client.calls[0]
        self.assertEqual(params, {
            "wallet": "prl1ptest",
            "worker": "mac",
            "pass": "x;d=20000",
            "agent": "pearl-apple-miner/1.0.3",
        })

    def test_explicit_password_can_include_pool_options(self):
        client = RecordingClient(
            "pool.example:1234", "prl1ptest", difficulty=20000,
            password="x;d=20000;t=25",
        )
        client.authorize()
        self.assertEqual(client.calls[0][1]["pass"], "x;d=20000;t=25")

    def test_worker_and_pass_are_optional(self):
        client = RecordingClient("pool.example:1234", "prl1ptest", agent="test-miner/1")
        client.authorize()
        self.assertEqual(client.calls[0][1], {
            "wallet": "prl1ptest",
            "agent": "test-miner/1",
        })

    def test_keepalive_uses_configure_after_idle(self):
        client = StratumPoolClient(
            "pool.example:1234", "prl1ptest", keepalive_interval=1,
        )
        sent = []
        client._authorized = True
        client._last_send = 0
        client._send = sent.append
        client.keepalive_if_idle()
        self.assertEqual(sent[0]["method"], "mining.configure")


if __name__ == "__main__":
    unittest.main()
