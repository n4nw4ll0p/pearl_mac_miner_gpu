"""Local-side block construction for Pearl.

Pure CPU code. No network. Takes parsed `getblocktemplate`
output + a mined PlainProof and produces the bytes to hand to `submitblock`.

The functions here are the same ones used by the original `live_miner.py` —
they are factored into their own module so the new controller can reuse them.
"""

from __future__ import annotations

from hashlib import sha256

from bitcoinutils.transactions import Transaction
from bitcoinutils.utils import encode_varint


PROOF_COMMITMENT_SIZE = 32
ZK_CERT_VERSION = 1

# Bitcoin HRPs — a Pearl miner pasting one of these by mistake mines to a
# script the Pearl chain happily accepts but whose private key is held on the
# Bitcoin chain. Reward effectively unspendable until the key is imported into
# a Pearl-aware wallet. Reject these explicitly so the user gets a clear error.
_REJECTED_HRPS = frozenset({"bc", "tb", "bcrt"})


def double_sha256(data: bytes) -> bytes:
    return sha256(sha256(data).digest()).digest()


def compute_merkle_root(tx_hashes_le: list[bytes]) -> bytes:
    """Bitcoin merkle root from a list of LE txids. Returns big-endian root."""
    if len(tx_hashes_le) == 1:
        return tx_hashes_le[0][::-1]
    cur = tx_hashes_le.copy()
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            a = cur[i]
            b = cur[i + 1] if i + 1 < len(cur) else cur[i]
            nxt.append(double_sha256(a + b))
        cur = nxt
    return cur[0][::-1]


def make_coinbase_tx(
    height: int,
    coinbase_value: int,
    mining_address: str,
    coinbase_aux_flags_hex: str | None,
    default_witness_commitment_hex: str | None,
) -> Transaction:
    """Build a coinbase tx paying `coinbase_value` to the given Pearl bech32m address.

    Accepts any HRP that isn't a known Bitcoin prefix (bc/tb/bcrt are rejected
    to guard against the operator pasting a Bitcoin taproot address — Pearl
    would happily mine to that script but the key is on the wrong chain).
    The output scriptPubKey is a v1 (taproot) 32-byte witness program.
    """
    from bitcoinutils.bech32 import Encoding, bech32_decode, convertbits
    from bitcoinutils.script import Script
    from bitcoinutils.transactions import TxInput, TxOutput, TxWitnessInput

    hrp, data, encoding = bech32_decode(mining_address)
    if hrp is None or encoding != Encoding.BECH32M or data[0] != 1:
        raise ValueError(f"invalid Taproot bech32m address: {mining_address}")
    if hrp in _REJECTED_HRPS:
        raise ValueError(
            f"address HRP {hrp!r} is a Bitcoin prefix — Pearl rewards paid to "
            f"this script would be unspendable from a Bitcoin wallet. Use a "
            f"Pearl bech32m address (HRP rprl/prl/...) instead of {mining_address!r}."
        )
    program_bits = convertbits(data[1:], 5, 8, False)
    if program_bits is None:
        raise ValueError(f"malformed taproot witness program in address: {mining_address}")
    program_bytes = bytes(program_bits)
    if len(program_bytes) != 32:
        raise ValueError("Taproot program must be 32 bytes")
    script_pubkey = Script.from_raw("5120" + program_bytes.hex())

    height_script_bytes = bytes.fromhex(Script([height]).to_hex())
    coinbase_script_bytes = height_script_bytes + b"\x00"
    if coinbase_aux_flags_hex:
        coinbase_script_bytes += bytes.fromhex(coinbase_aux_flags_hex)

    coinbase_input = TxInput(
        txid="0" * 64,
        txout_index=0xFFFFFFFF,
        script_sig=Script([coinbase_script_bytes.hex()]),
        sequence=b"\xff\xff\xff\xff",
    )
    outs = [TxOutput(coinbase_value, script_pubkey)]
    has_witness = default_witness_commitment_hex is not None
    if has_witness:
        witness_script = Script(["OP_RETURN", "aa21a9ed" + default_witness_commitment_hex])
        outs.append(TxOutput(0, witness_script))
    return Transaction(
        inputs=[coinbase_input],
        outputs=outs,
        locktime=b"\x00\x00\x00\x00",
        version=b"\x01\x00\x00\x00",
        has_segwit=has_witness,
        witnesses=[TxWitnessInput([bytes(32).hex()])] if has_witness else None,
    )


def build_zk_cert(header_bytes_76: bytes, public_data: bytes, proof_data: bytes) -> tuple[bytes, bytes, bytes]:
    """Pack the Plonky2 ZK proof into Pearl's certificate envelope.

    Returns (cert_bytes, header_with_commitment, header_hash). The header hash
    is the canonical block hash (double_sha256 of the 76-byte header followed
    by the 32-byte proof commitment).
    """
    version_bytes = ZK_CERT_VERSION.to_bytes(4, "little")
    proof_commitment = double_sha256(version_bytes + public_data)
    header_with_commitment = header_bytes_76 + proof_commitment
    assert len(header_with_commitment) == 76 + PROOF_COMMITMENT_SIZE
    header_hash = double_sha256(header_with_commitment)

    cert_bytes = (
        version_bytes
        + header_hash
        + public_data
        + len(proof_data).to_bytes(4, "little")
        + proof_data
    )
    return cert_bytes, header_with_commitment, header_hash


def assemble_block_bytes(
    cert_bytes: bytes,
    header_with_commitment: bytes,
    coinbase_tx_bytes: bytes,
    raw_other_txn_bytes: list[bytes],
) -> bytes:
    """Assemble the final submitblock payload: ZK cert | header+commit | varint(n) | txs."""
    all_tx_bytes = [coinbase_tx_bytes] + raw_other_txn_bytes
    return (
        cert_bytes
        + header_with_commitment
        + encode_varint(len(all_tx_bytes))
        + b"".join(all_tx_bytes)
    )
