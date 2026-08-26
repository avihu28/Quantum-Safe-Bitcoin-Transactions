#!/usr/bin/env python3
"""
Gold-standard consensus gate for the config_a builder.

Builds the config_a lock and runs it through REAL Bitcoin Core consensus code
(libbitcoinconsensus, via ../../QSB/qsb_verify). The puzzle CHECKSIGVERIFYs are
relaxed to OP_2DROP (identical opcode count + stack effect) because the ~2^46
hash-to-DER puzzle is unsolved here; the pinning bind and BOTH CHECKMULTISIGs
plus every HORS EQUALVERIFY stay REAL, so the off-by-one selection logic and the
SIGHASH_SINGLE-bug dummy recovery are exercised for real.

Modes:
  --witness legacy : replicate config_a/pipeline/qsb_pipeline.cmd_assemble exactly
                     (raw pool indices, dummy pubkeys recovered with z=1).
  --witness fixed  : model-derived indices + dummy pubkeys recovered with z=2**248.

Config Ar = ripemd160 puzzle, Config A = sha256 puzzle (both single-hash).
"""
import sys, struct, hashlib, subprocess, tempfile, os, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent          # config_a/
sys.path.insert(0, str(ROOT / "pipeline"))

# ── pure-python RIPEMD160 (OpenSSL3 dropped it) ──
def _rol(x,n): return ((x<<n)|(x>>(32-n)))&0xFFFFFFFF
_K1=(0,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E);_K2=(0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0)
_R1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13]
_R2=[5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11]
_S1=[11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6]
_S2=[8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11]
def _f(j,x,y,z):
    if j<16:return x^y^z
    if j<32:return (x&y)|(~x&z)
    if j<48:return (x|~y)^z
    if j<64:return (x&z)|(y&~z)
    return x^(y|~z)
def ripemd160(data):
    h=[0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0];msg=bytearray(data);ml=len(data)*8;msg.append(0x80)
    while len(msg)%64!=56:msg.append(0)
    msg+=struct.pack('<Q',ml)
    for off in range(0,len(msg),64):
        X=struct.unpack_from('<16I',msg,off);a,b,c,d,e=h;A,B,C,D,E=h
        for j in range(80):
            T=(a+_f(j,b,c,d)+X[_R1[j]]+_K1[j//16])&0xFFFFFFFF;T=(_rol(T,_S1[j])+e)&0xFFFFFFFF;a,e,d,c,b=e,d,_rol(c,10),b,T
            T=(A+_f(79-j,B,C,D)+X[_R2[j]]+_K2[j//16])&0xFFFFFFFF;T=(_rol(T,_S2[j])+E)&0xFFFFFFFF;A,E,D,C,B=E,D,_rol(C,10),B,T
        T=(h[1]+c+D)&0xFFFFFFFF;h[1]=(h[2]+d+E)&0xFFFFFFFF;h[2]=(h[3]+e+A)&0xFFFFFFFF;h[3]=(h[4]+a+B)&0xFFFFFFFF;h[4]=(h[0]+b+C)&0xFFFFFFFF;h[0]=T
    return struct.pack('<5I',*h)

import secp256k1 as S
S.ripemd160=ripemd160; S.hash160=lambda d: ripemd160(hashlib.sha256(d).digest())
import bitcoin_tx as bt
bt.ripemd160=ripemd160; bt.hash160=S.hash160
from bitcoin_tx import (QSBScriptBuilder, Transaction, TxIn, TxOut,
                        find_and_delete, push_data, push_number, _valid_small_r_values)
from secp256k1 import N, P, ecdsa_recover, ecdsa_verify, compress_pubkey, encode_der_sig

QV = "/home/tomer/workspace/QSB/qsb_verify/target/release/qsb_verify"

def parse_der(sig):
    if len(sig)<9 or sig[0]!=0x30: return None
    if sig[1]+3!=len(sig): return None
    i=2
    if sig[i]!=0x02: return None
    i+=1; rl=sig[i]; i+=1
    if rl==0 or i+rl>len(sig)-1 or sig[i]&0x80: return None
    if rl>1 and sig[i]==0 and not (sig[i+1]&0x80): return None
    r=int.from_bytes(sig[i:i+rl],'big'); i+=rl
    if sig[i]!=0x02: return None
    i+=1; sl=sig[i]; i+=1
    if sl==0 or i+sl!=len(sig)-1: return None
    return (r, int.from_bytes(sig[i:i+sl],'big'), sig[-1])

def recover_key(r,s,z):
    for rr in ([r]+([r+N] if r+N<P else [])):
        for rec in (0,1):
            pt=ecdsa_recover(rr,s,z,rec)
            if pt and ecdsa_verify(pt,z,r,s): return compress_pubkey(pt)
    return None

def relax(script):
    """Turn every puzzle CHECKSIGVERIFY (all but the 1st = pinning bind) into OP_2DROP."""
    out=bytearray(script); i=0; seen=0
    while i<len(out):
        op=out[i]
        if op==0: i+=1
        elif 1<=op<=75: i+=1+op
        elif op==0x4c: i+=2+out[i+1]
        elif op==0x4d: i+=3+(out[i+1]|out[i+2]<<8)
        else:
            if op==0xad:
                seen+=1
                if seen>=2: out[i]=0x6d      # OP_2DROP
            i+=1
    return bytes(out)

def run(config_name, cfg, witness_mode):
    b=QSBScriptBuilder(cfg['n'],cfg['t1s'],cfg['t1b'],cfg['t2s'],cfg['t2b'],hash_mode=cfg['hash_mode'])
    b.generate_keys()
    vr=_valid_small_r_values()[0]
    pin_sig=encode_der_sig(vr,1,0x01); sig_nonce=[encode_der_sig(vr,2,0x01), encode_der_sig(vr,3,0x01)]
    lock=b.build_full_script(pin_sig, sig_nonce[0], sig_nonce[1])
    lock_relaxed=relax(lock)

    QSB=1
    tx=Transaction(version=1, locktime=1234567)
    tx.add_input(TxIn(b'\x00'*32,0,b'',0xfffffffe))
    tx.add_input(TxIn(b'\x11'*32,0,b'',0x80000000))
    tx.add_output(TxOut(90000, bytes([0x00,0x14])+b'\x22'*20))
    t1=cfg['t1s']+cfg['t1b']; t2=cfg['t2s']+cfg['t2b']
    subs={0: sorted([3,17,42,66,88,101,119,140][:cfg['t1s']])+[9,29,49][:cfg['t1b']],
          1: sorted([5,20,55,70,90,110,130][:cfg['t2s']])+[15,45,75][:cfg['t2b']]}
    subs={0: subs[0][:t1], 1: subs[1][:t2]}

    z_for_dummy = 1 if witness_mode=='legacy' else (1<<248)
    if witness_mode=='fixed':
        iv=b.compute_witness_indices(subs)
    else:
        iv={0: list(reversed(subs[0]))[::-1], 1: list(reversed(subs[1]))[::-1]}  # raw indices

    def round_keys(R):
        sc=find_and_delete(lock_relaxed, sig_nonce[R])
        for i in subs[R]: sc=find_and_delete(sc, b.dummy_sigs[R][i])
        z=tx.sighash(QSB, sc, 0x01)
        pr,ps,_=parse_der(sig_nonce[R]); kn=recover_key(pr,ps,z)
        dp=[recover_key(*parse_der(b.dummy_sigs[R][i])[:2], z_for_dummy) for i in subs[R]]
        return kn,dp

    wit=b''
    for R in (1,0):
        ts=cfg['t1s'] if R==0 else cfg['t2s']
        kn,dp=round_keys(R)
        if not (kn and all(dp)): return (config_name, witness_mode, False, f"recover round {R}")
        wit+=push_data(b'\x02'+b'\x00'*32)+push_data(kn)
        for p in reversed(dp): wit+=push_data(p)
        for j in range(ts-1,-1,-1): wit+=push_data(b.hors_secrets[R][subs[R][j]])
        if witness_mode=='fixed':
            for j in range(len(iv[R])-1,-1,-1): wit+=push_number(iv[R][j])
        else:
            for idx in reversed(subs[R]): wit+=push_number(idx)
    psc=find_and_delete(lock_relaxed, pin_sig); zp=tx.sighash(QSB,psc,0x01)
    ppr,pps,_=parse_der(pin_sig); knp=recover_key(ppr,pps,zp)
    if not knp: return (config_name, witness_mode, False, "pin recover")
    wit+=push_data(b'\x02'+b'\x00'*32)+push_data(knp)
    tx.inputs[QSB].script_sig=wit

    ftx=Transaction(version=1, locktime=0)
    ftx.add_input(TxIn(b'\x33'*32,0,b'',0xffffffff))
    ftx.add_output(TxOut(100000, lock_relaxed))

    d=tempfile.mkdtemp()
    sp=os.path.join(d,'spend.hex'); fp=os.path.join(d,'fund.hex')
    open(sp,'w').write(tx.serialize().hex()); open(fp,'w').write(ftx.serialize().hex())
    r=subprocess.run([QV, sp, fp, "0", "1"], capture_output=True, text=True)
    ok = any("ALL_PRE_TAPROOT" in ln and "✅ SUCCESS" in ln for ln in r.stdout.splitlines())
    detail = "consensus-valid" if ok else "REJECTED by libbitcoinconsensus"
    if os.environ.get("QSB_DEBUG"):
        print("---- qsb_verify stdout ----"); print(r.stdout[-1200:]); print("---- stderr ----"); print(r.stderr[-400:])
    return (config_name, witness_mode, ok, detail)

CONFIGS = {
    'Ar': {'n':150,'t1s':8,'t1b':1,'t2s':7,'t2b':2,'hash_mode':'ripemd160'},
    'A' : {'n':150,'t1s':8,'t1b':1,'t2s':7,'t2b':2,'hash_mode':'sha256'},
}

def cfg_hm(c): return c['hash_mode']

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--witness', choices=['legacy','fixed'], default='fixed')
    ap.add_argument('--config', choices=list(CONFIGS)+['all'], default='all')
    a=ap.parse_args()
    names=list(CONFIGS) if a.config=='all' else [a.config]
    print(f"witness mode: {a.witness}\n")
    allok=True
    for nm in names:
        cn,wm,ok,det=run(nm, CONFIGS[nm], a.witness)
        allok &= ok
        print(f"  [{cn:2s} / {cfg_hm(CONFIGS[nm])}] {'✅' if ok else '❌'} {det}")
    sys.exit(0 if allok else 1)
