#!/usr/bin/env python3
"""
sign_tx.py — sign one or more SegWit (P2WPKH) inputs of a Bitcoin transaction
using a BIP39 seed phrase derived through BIP84 (m/84'/0'/0'/0/0).

Reads the seed phrase from stdin (one line, space-separated words).

Usage:
    cat | python3 sign_tx.py qsb_funding_unsigned.hex \
        --output qsb_funding_signed.hex \
        --input 0 --input-value 143534 \
        --expected-address bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj

For TX2 (input[0] only — input[1] already signed via QSB witness):
    cat | python3 sign_tx.py qsb_raw_tx.hex \
        --output qsb_raw_signed.hex \
        --input 0 --input-value 63402 \
        --expected-address bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj

Stdin: paste the 12 or 24 word mnemonic, then Ctrl-D (or pipe via cat).

Pure stdlib + the project's secp256k1 module. No pip required.
"""
import argparse
import hashlib
import hmac
import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from secp256k1 import (  # type: ignore
    point_mul, compress_pubkey, ecdsa_sign,
    encode_der_sig, hash160, N as CURVE_N,
)

G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)


# ─── BIP39: mnemonic → 64-byte binary seed ────────────────────────────
def mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
    """BIP39: PBKDF2-HMAC-SHA512(password=mnemonic, salt='mnemonic'+passphrase, 2048)."""
    mn = " ".join(mnemonic.lower().strip().split())
    salt = ("mnemonic" + passphrase).encode("utf-8")
    return hashlib.pbkdf2_hmac("sha512", mn.encode("utf-8"), salt, 2048, dklen=64)


# ─── BIP32: derive child keys ─────────────────────────────────────────
def bip32_master(seed: bytes) -> tuple[int, bytes]:
    """Master key from seed. Returns (privkey_int, chain_code)."""
    h = hmac.new(b"Bitcoin seed", seed, hashlib.sha512).digest()
    k = int.from_bytes(h[:32], "big")
    if k == 0 or k >= CURVE_N:
        raise ValueError("invalid master key (extremely improbable)")
    return k, h[32:]


def bip32_ckd_priv(parent_k: int, parent_cc: bytes, index: int) -> tuple[int, bytes]:
    """BIP32 child key derivation (private)."""
    if index >= 0x80000000:
        # hardened: data = 0x00 || ser256(k_parent) || ser32(i)
        data = b"\x00" + parent_k.to_bytes(32, "big") + struct.pack(">I", index)
    else:
        # normal: data = serP(K_parent) || ser32(i)
        parent_pub = point_mul(parent_k, G)
        data = compress_pubkey(parent_pub) + struct.pack(">I", index)
    h = hmac.new(parent_cc, data, hashlib.sha512).digest()
    il = int.from_bytes(h[:32], "big")
    if il >= CURVE_N:
        raise ValueError(f"invalid child key derivation at index {index}")
    child_k = (il + parent_k) % CURVE_N
    if child_k == 0:
        raise ValueError(f"child key is zero at index {index}")
    return child_k, h[32:]


def derive_bip84_path(seed: bytes, account: int = 0,
                      change: int = 0, index: int = 0) -> int:
    """Derive m/84'/0'/<account>'/<change>/<index>. Returns privkey_int."""
    H = 0x80000000  # hardened bit
    k, cc = bip32_master(seed)
    path = [84 + H, 0 + H, account + H, change, index]
    for i in path:
        k, cc = bip32_ckd_priv(k, cc, i)
    return k


# ─── bech32 encoding for address verification ────────────────────────
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def bech32_polymod(values: list[int]) -> int:
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GEN[i]
    return chk


def bech32_hrp_expand(hrp: str) -> list[int]:
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def bech32_create_checksum(hrp: str, data: list[int], spec: str) -> list[int]:
    const = 1 if spec == "bech32" else 0x2bc830a3  # bech32m
    polymod = bech32_polymod(bech32_hrp_expand(hrp) + data + [0]*6) ^ const
    return [(polymod >> 5*(5 - i)) & 31 for i in range(6)]


def convertbits(data, frombits, tobits, pad=True):
    acc, bits, ret = 0, 0, []
    maxv = (1 << tobits) - 1
    for v in data:
        acc = (acc << frombits) | v
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    return ret


def encode_p2wpkh_address(pubkey_hash: bytes, hrp: str = "bc") -> str:
    """SegWit v0 P2WPKH address (bech32, not bech32m)."""
    witver = 0
    data = [witver] + convertbits(pubkey_hash, 8, 5)
    chk = bech32_create_checksum(hrp, data, "bech32")
    combined = data + chk
    return hrp + "1" + "".join(BECH32_CHARSET[d] for d in combined)


# ─── Bitcoin tx parsing / serialization (segwit) ─────────────────────
def varint_encode(n: int) -> bytes:
    if n < 0xfd: return bytes([n])
    if n <= 0xffff: return b"\xfd" + n.to_bytes(2, "little")
    if n <= 0xffffffff: return b"\xfe" + n.to_bytes(4, "little")
    return b"\xff" + n.to_bytes(8, "little")


def varint_decode(b: bytes, off: int):
    f = b[off]
    if f < 0xfd: return f, off + 1
    if f == 0xfd: return int.from_bytes(b[off+1:off+3], "little"), off + 3
    if f == 0xfe: return int.from_bytes(b[off+1:off+5], "little"), off + 5
    return int.from_bytes(b[off+1:off+9], "little"), off + 9


def parse_tx(hex_str: str) -> dict:
    raw = bytes.fromhex(hex_str)
    o = 0
    version = int.from_bytes(raw[o:o+4], "little"); o += 4
    is_segwit = False
    if raw[o:o+2] == b"\x00\x01":
        is_segwit = True; o += 2
    n_in, o = varint_decode(raw, o)
    inputs = []
    for _ in range(n_in):
        prev = raw[o:o+32]; o += 32
        vout = int.from_bytes(raw[o:o+4], "little"); o += 4
        sl, o = varint_decode(raw, o)
        ss = raw[o:o+sl]; o += sl
        seq = int.from_bytes(raw[o:o+4], "little"); o += 4
        inputs.append({"prev": prev, "vout": vout, "scriptSig": ss, "sequence": seq})
    n_out, o = varint_decode(raw, o)
    outputs = []
    for _ in range(n_out):
        v = int.from_bytes(raw[o:o+8], "little"); o += 8
        sl, o = varint_decode(raw, o)
        s = raw[o:o+sl]; o += sl
        outputs.append({"value": v, "script": s})
    witnesses = [[] for _ in range(n_in)]
    if is_segwit:
        for i in range(n_in):
            n_w, o = varint_decode(raw, o)
            for _ in range(n_w):
                wl, o = varint_decode(raw, o)
                witnesses[i].append(raw[o:o+wl]); o += wl
    locktime = int.from_bytes(raw[o:o+4], "little"); o += 4
    return {"version": version, "is_segwit": is_segwit, "inputs": inputs,
            "outputs": outputs, "witnesses": witnesses, "locktime": locktime}


def serialize_tx(tx: dict, force_segwit: bool = True) -> bytes:
    has_witness = any(w for w in tx["witnesses"]) if tx.get("witnesses") else False
    out = struct.pack("<I", tx["version"])
    if force_segwit or has_witness:
        out += b"\x00\x01"
    out += varint_encode(len(tx["inputs"]))
    for i in tx["inputs"]:
        out += i["prev"] + struct.pack("<I", i["vout"])
        out += varint_encode(len(i["scriptSig"])) + i["scriptSig"]
        out += struct.pack("<I", i["sequence"])
    out += varint_encode(len(tx["outputs"]))
    for o in tx["outputs"]:
        out += struct.pack("<Q", o["value"])
        out += varint_encode(len(o["script"])) + o["script"]
    if force_segwit or has_witness:
        for w in tx["witnesses"]:
            out += varint_encode(len(w))
            for item in w:
                out += varint_encode(len(item)) + item
    out += struct.pack("<I", tx["locktime"])
    return out


def bip143_sighash(tx: dict, input_index: int, script_code: bytes,
                   amount: int, sighash_type: int = 0x01) -> bytes:
    """BIP143 SegWit sighash."""
    le = lambda n, w: n.to_bytes(w, "little")
    pp = b"".join(i["prev"] + le(i["vout"], 4) for i in tx["inputs"])
    hash_prevouts = hashlib.sha256(hashlib.sha256(pp).digest()).digest()
    seqs = b"".join(le(i["sequence"], 4) for i in tx["inputs"])
    hash_sequence = hashlib.sha256(hashlib.sha256(seqs).digest()).digest()
    outs = b""
    for o in tx["outputs"]:
        outs += le(o["value"], 8) + varint_encode(len(o["script"])) + o["script"]
    hash_outputs = hashlib.sha256(hashlib.sha256(outs).digest()).digest()
    inp = tx["inputs"][input_index]
    pre = (
        le(tx["version"], 4)
        + hash_prevouts + hash_sequence
        + inp["prev"] + le(inp["vout"], 4)
        + varint_encode(len(script_code)) + script_code
        + le(amount, 8)
        + le(inp["sequence"], 4)
        + hash_outputs
        + le(tx["locktime"], 4)
        + le(sighash_type, 4)
    )
    return hashlib.sha256(hashlib.sha256(pre).digest()).digest()


# ─── main signing logic ──────────────────────────────────────────────
def sign_p2wpkh_input(tx: dict, input_index: int, amount: int,
                      privkey_int: int, sighash_type: int = 0x01) -> None:
    """Sign one P2WPKH input in-place. Sets witness=[der_sig+ht, pubkey]."""
    pub_pt = point_mul(privkey_int, G)
    pubkey_compressed = compress_pubkey(pub_pt)
    pkh = hash160(pubkey_compressed)
    # P2WPKH scriptCode: OP_DUP OP_HASH160 <20-byte pkh> OP_EQUALVERIFY OP_CHECKSIG
    script_code = b"\x76\xa9\x14" + pkh + b"\x88\xac"
    z = bip143_sighash(tx, input_index, script_code, amount, sighash_type)
    z_int = int.from_bytes(z, "big")
    r, s = ecdsa_sign(privkey_int, z_int)
    der = encode_der_sig(r, s, sighash=sighash_type)
    tx["witnesses"][input_index] = [der, pubkey_compressed]
    # Clear scriptSig (segwit inputs have empty scriptSig)
    tx["inputs"][input_index]["scriptSig"] = b""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tx_hex_file", help="path to file containing the unsigned tx hex")
    ap.add_argument("--output", required=True, help="path to write signed tx hex")
    ap.add_argument("--input", type=int, action="append", required=True,
                    help="input index to sign (P2WPKH). May be specified multiple times.")
    ap.add_argument("--input-value", type=int, action="append", required=True,
                    help="value (sats) of the corresponding --input. Same count as --input.")
    ap.add_argument("--expected-address", default="bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj",
                    help="P2WPKH address that derived key MUST equal (sanity check)")
    ap.add_argument("--account", type=int, default=0, help="BIP84 account (default 0)")
    ap.add_argument("--change", type=int, default=0, help="BIP84 change (default 0 = receive chain)")
    ap.add_argument("--index", type=int, default=0, help="BIP84 address index (default 0)")
    ap.add_argument("--passphrase", default="",
                    help="BIP39 passphrase (default empty)")
    args = ap.parse_args()

    if len(args.input) != len(args.input_value):
        print("ERROR: --input and --input-value count must match", file=sys.stderr)
        sys.exit(2)

    # Read tx hex
    tx_hex = Path(args.tx_hex_file).read_text().strip()
    tx = parse_tx(tx_hex)
    print(f"  Parsed tx: version={tx['version']}, "
          f"inputs={len(tx['inputs'])}, outputs={len(tx['outputs'])}, "
          f"locktime={tx['locktime']}", file=sys.stderr)

    # Read mnemonic from stdin
    print("\nPaste your BIP39 mnemonic on one line, then press Enter:",
          file=sys.stderr, flush=True)
    mnemonic = sys.stdin.readline().strip()
    if not mnemonic:
        print("ERROR: empty mnemonic from stdin", file=sys.stderr)
        sys.exit(2)
    word_count = len(mnemonic.split())
    if word_count not in (12, 15, 18, 21, 24):
        print(f"ERROR: mnemonic word count {word_count} is not 12/15/18/21/24",
              file=sys.stderr)
        sys.exit(2)

    # Derive privkey
    seed = mnemonic_to_seed(mnemonic, args.passphrase)
    pk = derive_bip84_path(seed, account=args.account, change=args.change, index=args.index)

    # Sanity check: derived address matches expected
    pub = point_mul(pk, G)
    pub_c = compress_pubkey(pub)
    pkh = hash160(pub_c)
    derived_addr = encode_p2wpkh_address(pkh)
    if derived_addr != args.expected_address:
        print(f"\n✗ ADDRESS MISMATCH:", file=sys.stderr)
        print(f"   derived:  {derived_addr}", file=sys.stderr)
        print(f"   expected: {args.expected_address}", file=sys.stderr)
        print(f"   path:     m/84'/0'/{args.account}'/{args.change}/{args.index}",
              file=sys.stderr)
        print(f"   Either the mnemonic or the derivation path is wrong.", file=sys.stderr)
        sys.exit(3)
    print(f"  ✓ Derived address matches expected: {derived_addr}", file=sys.stderr)

    # Sign each requested input
    for idx, val in zip(args.input, args.input_value):
        if idx >= len(tx["inputs"]):
            print(f"ERROR: --input {idx} out of range", file=sys.stderr)
            sys.exit(2)
        sign_p2wpkh_input(tx, idx, val, pk)
        print(f"  ✓ Signed input[{idx}]  (amount={val} sats)", file=sys.stderr)

    # Serialize and write
    signed_raw = serialize_tx(tx, force_segwit=True)
    Path(args.output).write_text(signed_raw.hex())
    print(f"\n  Signed tx written to {args.output} ({len(signed_raw)} bytes)",
          file=sys.stderr)
    print(f"  txid: {hashlib.sha256(hashlib.sha256(serialize_tx_no_witness(tx)).digest()).digest()[::-1].hex()}",
          file=sys.stderr)


def serialize_tx_no_witness(tx: dict) -> bytes:
    """Serialize without witness for txid computation."""
    out = struct.pack("<I", tx["version"])
    out += varint_encode(len(tx["inputs"]))
    for i in tx["inputs"]:
        out += i["prev"] + struct.pack("<I", i["vout"])
        out += varint_encode(len(i["scriptSig"])) + i["scriptSig"]
        out += struct.pack("<I", i["sequence"])
    out += varint_encode(len(tx["outputs"]))
    for o in tx["outputs"]:
        out += struct.pack("<Q", o["value"])
        out += varint_encode(len(o["script"])) + o["script"]
    out += struct.pack("<I", tx["locktime"])
    return out


if __name__ == "__main__":
    main()
