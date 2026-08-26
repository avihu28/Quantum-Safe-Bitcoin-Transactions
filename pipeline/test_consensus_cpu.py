#!/usr/bin/env python3
"""
test_consensus_cpu.py — the pre-GPU gate.

Builds the CORRECTED lock (real keys), assembles a spending tx + witness with
REAL recovered keys / real HORS preimages / model-derived indices, and runs the
QSB input through a faithful LEGACY interpreter that does REAL ECDSA on every
signature — EXCEPT the unsolved hash-to-DER puzzle, which is relaxed (a puzzle
sig is a fresh hash, i.e. not valid DER; in production the ~2^46 search makes it
valid DER, here we skip only that one gate).

What this proves that the earlier structural test could NOT:
  • pinning: sig_nonce really verifies against the recovered key_nonce
  • rounds : the 9 dummy sigs (z=1) AND sig_nonce really verify inside
             CHECKMULTISIG, in the exact pubkey/sig ORDER the fix produces
  • HORS  : real HASH160(preimage) == commitment
  • the whole script ends TRUE under real crypto

Relaxed = only the puzzle DER-validity (unchanged by the fix, already proven on
chain via pinning). Run on CPU in well under a second.
"""
import sys, os, struct, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── pure-python RIPEMD160 (OpenSSL 3 dropped it) ──
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
S.hash160   = lambda d: ripemd160(hashlib.sha256(d).digest())
import bitcoin_tx as bt
bt.ripemd160 = ripemd160; bt.hash160 = S.hash160
from bitcoin_tx import (QSBScriptBuilder, Transaction, TxIn, TxOut,
                        find_and_delete, push_data, push_number)
from secp256k1 import (N, P, ecdsa_recover, ecdsa_verify, decompress_pubkey,
                       compress_pubkey, encode_der_sig)

def h160(x): return ripemd160(hashlib.sha256(x).digest())

def parse_der(sig):
    """Return (r,s,flag) if `sig` is valid DER (+trailing flag byte), else None."""
    if len(sig) < 9 or sig[0] != 0x30: return None
    tl = sig[1]
    if tl + 3 != len(sig): return None
    idx = 2
    if sig[idx] != 0x02: return None
    idx += 1; rl = sig[idx]; idx += 1
    if rl == 0 or idx+rl > len(sig)-1: return None
    if rl > 1 and sig[idx] == 0 and not (sig[idx+1] & 0x80): return None
    if sig[idx] & 0x80: return None
    r = int.from_bytes(sig[idx:idx+rl], 'big'); idx += rl
    if sig[idx] != 0x02: return None
    idx += 1; sl = sig[idx]; idx += 1
    if sl == 0 or idx+sl != len(sig)-1: return None
    s = int.from_bytes(sig[idx:idx+sl], 'big')
    return (r, s, sig[-1])

def recover_key(r, s, z):
    """Recover a pubkey that verifies (r,s) over z (try both recids / r,r+N)."""
    for rr in ([r] + ([r+N] if r+N < P else [])):
        for recid in (0, 1):
            pt = ecdsa_recover(rr, s, z, recid)
            if pt and ecdsa_verify(pt, z, r, s):
                return compress_pubkey(pt)
    return None

# ══════════════════════════════════════════════════════════════════
# 1. Build the corrected lock (Config A) with real keys
# ══════════════════════════════════════════════════════════════════
b = QSBScriptBuilder(150, 8, 1, 7, 2)
b.generate_keys()
# distinct SIGHASH_ALL anchors; r MUST be a valid on-curve x-coordinate so
# key recovery works. Same r, distinct s → 3 distinct minimal 9-byte sigs;
# flag 0x01 keeps them distinct from the dummies (flag 0x03).
from bitcoin_tx import _valid_small_r_values
vr = _valid_small_r_values()[0]
pin_sig      = encode_der_sig(vr, 1, 0x01)
sig_nonce    = [encode_der_sig(vr, 2, 0x01), encode_der_sig(vr, 3, 0x01)]
lock = b.build_full_script(pin_sig, sig_nonce[0], sig_nonce[1])
print(f"lock: {len(lock)} bytes")

# ══════════════════════════════════════════════════════════════════
# 2. Spending tx (2 inputs, 1 output).  QSB at input 1.  Arbitrary
#    locktime/sequence/subsets — no search needed (puzzle relaxed).
# ══════════════════════════════════════════════════════════════════
QSB = 1
tx = Transaction(version=1, locktime=1234567)
tx.add_input(TxIn(b'\x00'*32, 0, b'', 0xfffffffe))            # helper (input 0)
tx.add_input(TxIn(b'\x11'*32, 0, b'', 0x80000000))            # QSB    (input 1)
tx.add_output(TxOut(90000, bytes([0x00,0x14])+b'\x22'*20))     # 1 output (SIGHASH_SINGLE bug at input 1)
subs = {0: sorted([3,17,42,66,88,101,119,140])+[9],
        1: sorted([5,20,55,70,90,110,130])+[15,45]}
idxvals = b.compute_witness_indices(subs)

# ══════════════════════════════════════════════════════════════════
# 3. Assemble the witness with REAL recovered keys / preimages
# ══════════════════════════════════════════════════════════════════
def recover_round_keys(R):
    sc = find_and_delete(lock, sig_nonce[R])
    for i in subs[R]:
        sc = find_and_delete(sc, b.dummy_sigs[R][i])
    z = tx.sighash(QSB, sc, 0x01)
    pr, ps, _ = parse_der(sig_nonce[R])
    key_nonce = recover_key(pr, ps, z)
    dummy_pub = []
    for i in subs[R]:
        dr, ds, _ = parse_der(b.dummy_sigs[R][i])
        dummy_pub.append(recover_key(dr, ds, 1 << 248))  # z=2**248 (corrected SIGHASH_SINGLE bug)
    return key_nonce, dummy_pub

witness = b''
for R in (1, 0):
    ts = b.t1_signed if R == 0 else b.t2_signed
    key_nonce, dummy_pub = recover_round_keys(R)
    assert key_nonce and all(dummy_pub), f"recovery failed round {R}"
    witness += push_data(b'\x02'+b'\x00'*32)          # key_puzzle (relaxed → unused)
    witness += push_data(key_nonce)                    # key_nonce (REAL)
    for pub in reversed(dummy_pub): witness += push_data(pub)
    for j in range(ts-1, -1, -1): witness += push_data(b.hors_secrets[R][subs[R][j]])
    for j in range(len(idxvals[R])-1, -1, -1): witness += push_number(idxvals[R][j])
# pinning
pin_sc = find_and_delete(lock, pin_sig)
z_pin = tx.sighash(QSB, pin_sc, 0x01)
ppr, pps, _ = parse_der(pin_sig)
key_nonce_pin = recover_key(ppr, pps, z_pin)
assert key_nonce_pin, "pinning recovery failed"
witness += push_data(b'\x02'+b'\x00'*32)               # key_puzzle_pin (relaxed)
witness += push_data(key_nonce_pin)

# ══════════════════════════════════════════════════════════════════
# 4. Interpret with REAL ECDSA (relax only the unsolved puzzle)
# ══════════════════════════════════════════════════════════════════
def toks(sc):
    i=0
    while i<len(sc):
        op=sc[i]
        if op==0: yield('p',b''); i+=1
        elif 1<=op<=75: yield('p',sc[i+1:i+1+op]); i+=1+op
        elif op==0x4c: k=sc[i+1]; yield('p',sc[i+2:i+2+k]); i+=2+k
        elif op==0x4d: k=sc[i+1]|sc[i+2]<<8; yield('p',sc[i+3:i+3+k]); i+=3+k
        elif 0x51<=op<=0x60: yield('p',bytes([op-0x50])); i+=1
        else: yield('o',op); i+=1
def di(x):
    if not x: return 0
    v=int.from_bytes(x,'little')
    if x[-1]&0x80: v&=(1<<(8*len(x)))-1-(0x80<<(8*(len(x)-1))); v=-v
    return v
def ei(v):
    if v==0: return b''
    o=bytearray(); y=abs(v)
    while y: o.append(y&0xff); y>>=8
    if o[-1]&0x80: o.append(0)
    return bytes(o)

def checksig_real(sig, pubkey, all_check_sigs):
    """REAL ecdsa verify over the legacy sighash. Relax if sig isn't valid DER
    (an unsolved puzzle sig) — that's the only relaxation."""
    p = parse_der(sig)
    if p is None:
        return ('relaxed', True)          # unsolved hash-to-DER puzzle → skip only this
    r, s, flag = p
    sc = lock
    for cs in all_check_sigs: sc = find_and_delete(sc, cs)
    z = tx.sighash(QSB, sc, flag)          # z=1 automatically for SIGHASH_SINGLE @ input1
    try: pt = decompress_pubkey(pubkey)
    except Exception: return ('real', False)
    return ('real', ecdsa_verify(pt, z, r, s))

st = [v for k,v in toks(witness)]
hors=0; cms=0; real_sig=0; relaxed=0; fail=None
for k,v in toks(lock):
    if k=='p': st.append(v); continue
    op=v
    try:
        if op==0x7a: nn=di(st.pop()); st.append(st.pop(-1-nn))
        elif op==0x76: st.append(st[-1])
        elif op==0x78: st.append(st[-2])
        elif op==0x7c: st[-1],st[-2]=st[-2],st[-1]
        elif op==0x93: y=di(st.pop());x=di(st.pop());st.append(ei(x+y))
        elif op==0xa3: y=di(st.pop());x=di(st.pop());st.append(ei(min(x,y)))
        elif op==0xa9: st.append(h160(st.pop()))
        elif op==0xa8: st.append(hashlib.sha256(st.pop()).digest())
        elif op==0xa6: st.append(ripemd160(st.pop()))
        elif op==0x88:                                   # OP_EQUALVERIFY = HORS
            a=st.pop(); c=st.pop()
            if a==c: hors+=1
            elif fail is None: fail=('HORS',a.hex()[:12],c.hex()[:12])
        elif op==0xad:                                   # OP_CHECKSIGVERIFY
            pub=st.pop(); sig=st.pop()
            kind,ok=checksig_real(sig, pub, [sig])
            if kind=='real': real_sig+=1
            else: relaxed+=1
            if not ok and fail is None: fail=('CHECKSIG',sig.hex()[:16],pub.hex()[:12])
        elif op==0xae:                                   # OP_CHECKMULTISIG (REAL, ordered)
            nk=di(st.pop()); pubs=[st.pop() for _ in range(nk)][::-1]
            ns=di(st.pop()); sigs=[st.pop() for _ in range(ns)][::-1]
            st.pop()                                      # the OP_0 dummy
            allcs=[s for s in sigs if parse_der(s)]       # F&D all real sigs in this CMS
            si=0; matched=0
            for sig in sigs:
                while si<nk:
                    kind,ok=checksig_real(sig, pubs[si], allcs)
                    if ok: matched+=1; si+=1; break
                    si+=1
                else: break
                if kind=='real': real_sig+=1
                else: relaxed+=1
            good = (matched==ns)
            st.append(b'\x01' if good else b''); cms+=1
            if not good and fail is None: fail=('CMS', f'{matched}/{ns}', '')
    except Exception as e:
        fail=('crash',str(e),''); break

top = st[-1] if st else b''
truthy = any(x!=0 for x in top)
exp_hors = b.t1_signed + b.t2_signed
print(f"HORS: {hors}/{exp_hors}   CHECKMULTISIGs: {cms}/2   real ECDSA checks: {real_sig}   relaxed(puzzle): {relaxed}")
print(f"final stack TRUE: {truthy}")
if fail: print("FAIL:", fail)
ok = (hors==exp_hors and cms==2 and truthy and fail is None)
print("\nCPU CONSENSUS TEST: " + ("PASS ✅  (real signatures verify; only the puzzle was relaxed)" if ok
                                  else "FAIL ❌"))
sys.exit(0 if ok else 1)
