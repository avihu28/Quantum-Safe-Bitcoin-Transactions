#!/usr/bin/env python3
"""
test_assemble_recovery.py — `assemble` must recover what the lock verifies
and what the v16 GPU kernels accept.

1. puzzle_hash() computes the same bytes as the hash opcode the lock runs,
   for each single-hash config (A = ripemd160, S = sha256). OP_RIPEMD160
   hashes the 33-byte key directly, not HASH160.
2. recover_pubkeys() also tries R.x = r + N. Consensus checks R.x mod N == r,
   and the v16 kernels accept a puzzle sig when r OR r + N is on the curve
   (gpu_der_r_on_curve). Every key it yields verifies; libsecp256k1
   (coincurve) cross-checks it when installed.
3. find_key_nonce() accepts only a puzzle hash the lock computes, so an
   SHA-256(SHA-256) "hit" is refused for a single-SHA-256 lock.
4. End to end: `setup` (configs A and S) -> `assemble` -> the scriptSig it
   writes satisfies the lock in a legacy interpreter that runs REAL ECDSA on
   every signature, including puzzle sigs whose R.x must be r + N.

The ~2^-46 hash-to-DER event cannot be produced on a CPU. Tests 3 and 4 stand
in for it by declaring the exact digests the lock will compute "valid DER"
and mapping them to chosen (r, s) (the same idea as the kernels' easy mode).
Nothing else is relaxed. Run: python3 test_assemble_recovery.py  (~10 s, CPU)
"""
import contextlib, hashlib, io, json, os, random, struct, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── pure-python RIPEMD160 (OpenSSL 3 may not ship it; same as test_consensus_cpu) ──
def _rol(x,n): return ((x<<n)|(x>>(32-n)))&0xFFFFFFFF
_K1=(0,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E); _K2=(0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0)
_R1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13]
_R2=[5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11]
_S1=[11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6]
_S2=[8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11]
def _f(j,x,y,z):
    if j<16: return x^y^z
    if j<32: return (x&y)|(~x&z)
    if j<48: return (x|~y)^z
    if j<64: return (x&z)|(y&~z)
    return x^(y|~z)
def ripemd160(data):
    h=[0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0]
    msg=bytearray(data); ml=len(data)*8; msg.append(0x80)
    while len(msg)%64!=56: msg.append(0)
    msg+=struct.pack('<Q',ml)
    for off in range(0,len(msg),64):
        X=struct.unpack_from('<16I',msg,off); a,b,c,d,e=h; A,B,C,D,E=h
        for j in range(80):
            T=(a+_f(j,b,c,d)+X[_R1[j]]+_K1[j//16])&0xFFFFFFFF; T=(_rol(T,_S1[j])+e)&0xFFFFFFFF
            a,e,d,c,b=e,d,_rol(c,10),b,T
            T=(A+_f(79-j,B,C,D)+X[_R2[j]]+_K2[j//16])&0xFFFFFFFF; T=(_rol(T,_S2[j])+E)&0xFFFFFFFF
            A,E,D,C,B=E,D,_rol(C,10),B,T
        T=(h[1]+c+D)&0xFFFFFFFF; h[1]=(h[2]+d+E)&0xFFFFFFFF; h[2]=(h[3]+e+A)&0xFFFFFFFF
        h[3]=(h[4]+a+B)&0xFFFFFFFF; h[4]=(h[0]+b+C)&0xFFFFFFFF; h[0]=T
    return struct.pack('<5I',*h)
assert ripemd160(b'abc').hex()=='8eb208f7e05d987a9b044a8e98c6b087f15a0bfc'

import secp256k1 as S
S.ripemd160 = ripemd160
S.hash160 = lambda d: ripemd160(hashlib.sha256(d).digest())
import bitcoin_tx as bt
bt.ripemd160 = ripemd160; bt.hash160 = S.hash160
import qsb_pipeline as qp
qp.ripemd160 = ripemd160; qp.hash160 = S.hash160

from bitcoin_tx import QSBScriptBuilder, Transaction, TxIn, find_and_delete, _valid_small_r_values
from secp256k1 import (N, P, G, point_mul, compress_pubkey, decompress_pubkey,
                       ecdsa_recover, ecdsa_verify, encode_der_sig)

def sha256(b): return hashlib.sha256(b).digest()

# What each hash opcode computes in Bitcoin's interpreter.
OPCODE_HASH = {0xa6: ripemd160,                          # OP_RIPEMD160
               0xa8: sha256,                             # OP_SHA256
               0xa9: lambda x: ripemd160(sha256(x))}     # OP_HASH160

def opcodes(script):
    """Opcodes of a script, skipping push payloads."""
    i, out = 0, []
    while i < len(script):
        op = script[i]
        if 1 <= op <= 75: i += 1 + op; continue
        if op == 0x4c: i += 2 + script[i+1]; continue
        if op == 0x4d: i += 3 + (script[i+1] | script[i+2] << 8); continue
        out.append(op); i += 1
    return out

def x_on_curve(x):
    return x < P and pow((pow(x, 3, P) + 7) % P, (P - 1) // 2, P) == 1

def det_int(tag, mod):
    return int.from_bytes(sha256(tag.encode()), 'big') % mod

# ══════════════════════════════════════════════════════════════════
# 1. puzzle_hash() == what the lock's puzzle opcode computes
# ══════════════════════════════════════════════════════════════════
vr = _valid_small_r_values()[0]
pin_sig = encode_der_sig(vr, 1, 0x01)
nonces = [encode_der_sig(vr, 2, 0x01), encode_der_sig(vr, 3, 0x01)]
CONFIGS = {'A': (150, 8, 1, 7, 2, 'ripemd160', 0xa6),
           'S': (140, 8, 1, 7, 2, 'sha256',    0xa8)}
for name, (n, t1s, t1b, t2s, t2b, mode, op) in CONFIGS.items():
    b = QSBScriptBuilder(n, t1s, t1b, t2s, t2b, hash_mode=mode)
    b.generate_keys()
    pin_ops = opcodes(b.build_pinning_script(pin_sig))
    assert pin_ops == [0x78, 0xad, op, 0x7c, 0xad], (name, pin_ops)   # OVER CSV <hash> SWAP CSV
    lock = b.build_full_script(pin_sig, nonces[0], nonces[1])
    assert opcodes(lock).count(op) == 3, name     # pinning + one puzzle per round
    for k in (1, 7, det_int('key', N)):
        key = compress_pubkey(point_mul(k, G))
        assert qp.puzzle_hash(key, mode)[0] == OPCODE_HASH[op](key), \
            f"config {name}: puzzle_hash({mode}) differs from opcode 0x{op:02x}"
    print(f"[OK] config {name}: puzzle_hash({mode}) matches the lock's opcode 0x{op:02x}")

# ══════════════════════════════════════════════════════════════════
# 2. recover_pubkeys() covers R.x = r + N
# ══════════════════════════════════════════════════════════════════
r = next(x for x in range(1, 1 << 16) if not x_on_curve(x) and x_on_curve(x + N))
s = det_int('s', N // 2) or 1                    # low-S, so libsecp256k1 accepts it too
z = det_int('z', N)
assert all(ecdsa_recover(r, s, z, f) is None for f in (0, 1)), "r itself should be off-curve"
keys = list(qp.recover_pubkeys(r, s, z))
assert len(keys) == 2 and all(plus_n for _, _, plus_n in keys), keys
for kb, _, _ in keys:
    assert ecdsa_verify(decompress_pubkey(kb), z, r, s)
try:
    import coincurve
except ImportError:
    coincurve = None
if coincurve:
    der = encode_der_sig(r, s)[:-1]              # drop the sighash byte
    for kb, _, _ in keys:
        assert coincurve.PublicKey(kb).verify(der, z.to_bytes(32, 'big'), hasher=None)
print(f"[OK] r={r}: off-curve, recovered via r+N; {len(keys)} keys verify"
      + (" (python + libsecp256k1)" if coincurve else " (python)"))

# an ordinary on-curve r keeps working and is labelled as such
k_ok = list(qp.recover_pubkeys(vr, s, z))
assert k_ok and not k_ok[0][2]
print(f"[OK] on-curve r={vr}: recovered via r")

# ══════════════════════════════════════════════════════════════════
# 3. find_key_nonce() accepts only the hash the lock computes
# ══════════════════════════════════════════════════════════════════
k0 = k_ok[0][0]
orig = qp.is_valid_der_sig
try:
    def only(target):
        qp.is_valid_der_sig = lambda h: h == target
    only(ripemd160(k0))
    got = qp.find_key_nonce(vr, s, z, 'ripemd160')
    assert got and got[0] == k0 and got[1] == ripemd160(k0), got
    only(sha256(k0))
    got = qp.find_key_nonce(vr, s, z, 'sha256')
    assert got and got[0] == k0 and got[1] == sha256(k0), got
    only(sha256(sha256(k0)))                     # a hash_choice=1 hit
    assert qp.find_key_nonce(vr, s, z, 'sha256') is None
    only(ripemd160(sha256(k0)))                  # HASH160 is not what OP_RIPEMD160 computes
    assert qp.find_key_nonce(vr, s, z, 'ripemd160') is None
finally:
    qp.is_valid_der_sig = orig
print("[OK] find_key_nonce: ripemd160/sha256 hits accepted, SHA-256² and HASH160 refused")

# ══════════════════════════════════════════════════════════════════
# 4. setup -> assemble -> interpret the spend with REAL ECDSA
# ══════════════════════════════════════════════════════════════════
def tokens(sc):
    i = 0
    while i < len(sc):
        op = sc[i]
        if op == 0: yield ('p', b''); i += 1
        elif 1 <= op <= 75: yield ('p', sc[i+1:i+1+op]); i += 1 + op
        elif op == 0x4c: k = sc[i+1]; yield ('p', sc[i+2:i+2+k]); i += 2 + k
        elif op == 0x4d: k = sc[i+1] | sc[i+2] << 8; yield ('p', sc[i+3:i+3+k]); i += 3 + k
        elif 0x51 <= op <= 0x60: yield ('p', bytes([op - 0x50])); i += 1
        else: yield ('o', op); i += 1
def dec_num(x):
    if not x: return 0
    v = int.from_bytes(x, 'little')
    if x[-1] & 0x80: v &= (1 << (8*len(x))) - 1 - (0x80 << (8*(len(x)-1))); v = -v
    return v
def enc_num(v):
    if v == 0: return b''
    o = bytearray(); y = abs(v)
    while y: o.append(y & 0xff); y >>= 8
    if o[-1] & 0x80: o.append(0)
    return bytes(o)

def first_key(r, s, z):
    for f in (0, 1):
        pt = ecdsa_recover(r, s, z, f)
        if pt: return compress_pubkey(pt)

def run_end_to_end(config, puzzle_r):
    """Returns (real ECDSA checks passed, key_puzzle labels printed by assemble)."""
    cwd = os.getcwd(); os.chdir(tempfile.mkdtemp())
    orig_valid, orig_parse = qp.is_valid_der_sig, qp.parse_der
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            qp.cmd_setup(types.SimpleNamespace(config=config, seed=424242))
        st = json.load(open(qp.STATE_FILE))
        lock, n = bytes.fromhex(st['full_script_hex']), st['n']
        puzzle_op = OPCODE_HASH[CONFIGS[config][6]]
        rng = random.Random(7)
        subs = [sorted(rng.sample(range(n), st['t1s'] + st['t1b'])),
                sorted(rng.sample(range(n), st['t2s'] + st['t2b']))]
        LT, SEQ, TXID = 500000123, 0xfffffffe, 'ab' * 32
        tx = Transaction(version=1, locktime=LT)             # same shape as cmd_assemble
        tx.add_input(TxIn(bytes.fromhex(TXID)[::-1], 0, b'', SEQ))

        # The three puzzle digests the lock will compute, declared "valid DER".
        target = {}
        z = tx.sighash(0, find_and_delete(lock, bytes.fromhex(st['pin_sig'])), 0x01)
        target[puzzle_op(first_key(st['pin_r'], st['pin_s'], z))] = (puzzle_r, 11111)
        for ri in range(2):
            rs = st['round_sigs'][ri]
            sc = find_and_delete(lock, bytes.fromhex(rs['sig']))
            for i in subs[ri]:
                sc = find_and_delete(sc, bytes.fromhex(st['dummy_sigs'][ri][i]))
            kn = first_key(rs['r'], rs['s'], tx.sighash(0, sc, 0x01))
            target[puzzle_op(kn)] = (puzzle_r, 22222 + ri)
        assert len(target) == 3
        qp.is_valid_der_sig = lambda h: h in target
        qp.parse_der = lambda b: target[b] if b in target else orig_parse(b)

        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            qp.cmd_assemble(types.SimpleNamespace(
                locktime=LT, sequence=SEQ, version=1, funding_txid=TXID,
                funding_vout=0, funding_value=100000,
                round1=','.join(map(str, subs[0])), round2=','.join(map(str, subs[1]))))
        assert os.path.exists('qsb_raw_tx.hex'), \
            [l.strip() for l in log.getvalue().splitlines() if 'ERROR' in l]
        raw = bytes.fromhex(open('qsb_raw_tx.hex').read())
        i = 41; L = raw[i]; i += 1
        if L == 0xfd: L = raw[i] | raw[i+1] << 8; i += 2
        script_sig = raw[i:i+L]
        assert raw[i+L:] == struct.pack('<I', SEQ) + b'\x00' + struct.pack('<I', LT)

        def checksig(sig, pub, cs):
            if sig in target: r, s = target[sig]
            else:
                r, s = orig_parse(sig)
                if r is None: return False
            sc = lock
            for c in cs: sc = find_and_delete(sc, c)
            try: pt = decompress_pubkey(pub)
            except ValueError: return False
            return ecdsa_verify(pt, tx.sighash(0, sc, sig[-1]), r, s)

        stk = [v for _, v in tokens(script_sig)]
        checks = hors = cms = 0
        for kind, v in tokens(lock):
            if kind == 'p': stk.append(v); continue
            if v == 0x7a: m = dec_num(stk.pop()); stk.append(stk.pop(-1-m))
            elif v == 0x76: stk.append(stk[-1])
            elif v == 0x78: stk.append(stk[-2])
            elif v == 0x7c: stk[-1], stk[-2] = stk[-2], stk[-1]
            elif v == 0x93: b_ = dec_num(stk.pop()); stk.append(enc_num(dec_num(stk.pop()) + b_))
            elif v == 0xa3: b_ = dec_num(stk.pop()); stk.append(enc_num(min(dec_num(stk.pop()), b_)))
            elif v in OPCODE_HASH: stk.append(OPCODE_HASH[v](stk.pop()))
            elif v == 0x88:                                  # OP_EQUALVERIFY (HORS)
                assert stk.pop() == stk.pop(), "HORS preimage mismatch"; hors += 1
            elif v == 0xad:                                  # OP_CHECKSIGVERIFY
                pub = stk.pop(); sig = stk.pop()
                assert checksig(sig, pub, [sig]), f"CHECKSIGVERIFY failed ({config})"
                checks += 1
            elif v == 0xae:                                  # OP_CHECKMULTISIG, in order
                nk = dec_num(stk.pop()); pubs = [stk.pop() for _ in range(nk)][::-1]
                ns = dec_num(stk.pop()); sigs = [stk.pop() for _ in range(ns)][::-1]
                stk.pop()                                    # the OP_0 dummy
                si = matched = 0
                for sig in sigs:
                    while si < nk:
                        si += 1
                        if checksig(sig, pubs[si-1], sigs): matched += 1; break
                assert matched == ns, f"CHECKMULTISIG {matched}/{ns} ({config})"
                checks += matched; cms += 1; stk.append(b'\x01')
            else:
                raise AssertionError(f"unhandled opcode 0x{v:02x}")
        assert cms == 2 and hors == st['t1s'] + st['t2s'] and stk and any(stk[-1])
        labels = sorted({l.split('(')[1].split(')')[0]
                         for l in log.getvalue().splitlines() if 'key_puzzle:' in l})
        return checks, labels
    finally:
        qp.is_valid_der_sig, qp.parse_der = orig_valid, orig_parse
        os.chdir(cwd)

for config in ('A', 'S'):
    checks, labels = run_end_to_end(config, r)       # r from test 2: needs R.x = r + N
    assert labels and all(l.endswith('r+N') for l in labels), labels
    print(f"[OK] config {config}: assembled spend satisfies the lock, "
          f"{checks} real ECDSA checks, key_puzzle via r+N")

print("\nALL PASS")
