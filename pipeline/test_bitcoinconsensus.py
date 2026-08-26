#!/usr/bin/env python3
"""
Gold-standard gate: run the CORRECTED lock through REAL Bitcoin Core consensus
code (libbitcoinconsensus, via ../../QSB/qsb_verify). Confirms our legacy
sighash == Core's — the one thing our own interpreter can't self-certify, and
the exact gap that could silently burn funding + GPU money.

Puzzle relaxed (each puzzle CHECKSIGVERIFY -> OP_2DROP, identical opcode count &
stack effect); pinning-bind + both CHECKMULTISIGs stay REAL. Emits funding+
spending tx hex and invokes qsb_verify on the QSB input.
"""
import sys, struct, hashlib, subprocess, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── pure-python RIPEMD160 (OpenSSL3) ──
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
                        find_and_delete, push_data, push_number, serialize_varint, _valid_small_r_values)
from secp256k1 import N, P, ecdsa_recover, ecdsa_verify, compress_pubkey, encode_der_sig

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

# ── build corrected lock (valid on-curve anchors) ──
b=QSBScriptBuilder(150,8,1,7,2); b.generate_keys()
vr=_valid_small_r_values()[0]
pin_sig=encode_der_sig(vr,1,0x01); sig_nonce=[encode_der_sig(vr,2,0x01), encode_der_sig(vr,3,0x01)]
lock=b.build_full_script(pin_sig, sig_nonce[0], sig_nonce[1])

# ── relax puzzle: keep 1st CHECKSIGVERIFY (pinning bind), turn the other 3 into OP_2DROP ──
def relax(script):
    out=bytearray(script); i=0; seen=0
    while i<len(out):
        op=out[i]
        if op==0: i+=1
        elif 1<=op<=75: i+=1+op
        elif op==0x4c: i+=2+out[i+1]
        elif op==0x4d: i+=3+(out[i+1]|out[i+2]<<8)
        else:
            if op==0xad:                       # OP_CHECKSIGVERIFY
                seen+=1
                if seen>=2: out[i]=0x6d          # OP_2DROP (relax the 3 puzzle checks)
            i+=1
    return bytes(out)
lock_relaxed=relax(lock)
print(f"lock: {len(lock)} bytes; relaxed 3 puzzle checks -> same length {len(lock_relaxed)}")

# ── spending tx (QSB @ input 1), arbitrary subsets (puzzle relaxed) ──
QSB=1
tx=Transaction(version=1, locktime=1234567)
tx.add_input(TxIn(b'\x00'*32,0,b'',0xfffffffe))
tx.add_input(TxIn(b'\x11'*32,0,b'',0x80000000))
tx.add_output(TxOut(90000, bytes([0x00,0x14])+b'\x22'*20))
subs={0:sorted([3,17,42,66,88,101,119,140])+[9], 1:sorted([5,20,55,70,90,110,130])+[15,45]}
iv=b.compute_witness_indices(subs)

def round_keys(R):
    sc=find_and_delete(lock_relaxed, sig_nonce[R])
    for i in subs[R]: sc=find_and_delete(sc, b.dummy_sigs[R][i])
    z=tx.sighash(QSB, sc, 0x01)
    pr,ps,_=parse_der(sig_nonce[R]); kn=recover_key(pr,ps,z)
    dp=[recover_key(*parse_der(b.dummy_sigs[R][i])[:2],1<<248) for i in subs[R]]
    return kn,dp

wit=b''
for R in (1,0):
    ts=b.t1_signed if R==0 else b.t2_signed
    kn,dp=round_keys(R); assert kn and all(dp), f"recover round {R}"
    wit+=push_data(b'\x02'+b'\x00'*32)+push_data(kn)
    for p in reversed(dp): wit+=push_data(p)
    for j in range(ts-1,-1,-1): wit+=push_data(b.hors_secrets[R][subs[R][j]])
    for j in range(len(iv[R])-1,-1,-1): wit+=push_number(iv[R][j])
psc=find_and_delete(lock_relaxed, pin_sig); zp=tx.sighash(QSB,psc,0x01)
ppr,pps,_=parse_der(pin_sig); knp=recover_key(ppr,pps,zp); assert knp,"pin recover"
wit+=push_data(b'\x02'+b'\x00'*32)+push_data(knp)
tx.inputs[QSB].script_sig=wit

# ── funding tx: vout 0 = relaxed lock ──
ftx=Transaction(version=1, locktime=0)
ftx.add_input(TxIn(b'\x33'*32,0,b'',0xffffffff))
ftx.add_output(TxOut(100000, lock_relaxed))

QV="/home/tomer/workspace/QSB/qsb_verify/target/release/qsb_verify"
def core_check(lockspk, witness_ss, label):
    ft=Transaction(1,0); ft.add_input(TxIn(b'\x33'*32,0,b'',0xffffffff)); ft.add_output(TxOut(100000,lockspk))
    st=Transaction(1,1234567); st.add_input(TxIn(b'\x00'*32,0,b'',0xfffffffe))
    st.add_input(TxIn(b'\x11'*32,0,witness_ss,0x80000000)); st.add_output(TxOut(90000, bytes([0x00,0x14])+b'\x22'*20))
    dd=tempfile.mkdtemp(); s=os.path.join(dd,'s'); f=os.path.join(dd,'f')
    open(s,'w').write(st.serialize().hex()); open(f,'w').write(ft.serialize().hex())
    r=subprocess.run([QV,s,f,"0","1"],capture_output=True,text=True)
    ok = "✅ SUCCESS" in r.stdout
    line=[l for l in r.stdout.splitlines() if "NONE" in l]
    print(f"  [{label}] Core VERIFY_NONE: {'✅ SUCCESS' if ok else '❌ '+(line[0].split('...')[-1].strip() if line else 'ERR')}")
    return ok

# --- DIAGNOSTIC 1: my own interpreter on the SAME relaxed lock ---
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
from secp256k1 import decompress_pubkey
def csig(sig,pub,allcs):
    p=parse_der(sig)
    if not p: return True
    r,s,fl=p; sc=lock_relaxed
    for cs in allcs: sc=find_and_delete(sc,cs)
    z=tx.sighash(QSB,sc,fl)
    try: return ecdsa_verify(decompress_pubkey(pub),z,r,s)
    except Exception: return False
st=[v for k,v in toks(wit)]; hp=cms=0; peak=len(st); fail=None
for k,v in toks(lock_relaxed):
    if k=='p': st.append(v); peak=max(peak,len(st)); continue
    op=v
    if op==0x7a: nn=di(st.pop()); st.append(st.pop(-1-nn))
    elif op==0x76: st.append(st[-1])
    elif op==0x78: st.append(st[-2])
    elif op==0x7c: st[-1],st[-2]=st[-2],st[-1]
    elif op==0x6d: st.pop(); st.pop()             # OP_2DROP (relaxed puzzle)
    elif op==0x93: y=di(st.pop());x=di(st.pop());st.append(ei(x+y))
    elif op==0xa3: y=di(st.pop());x=di(st.pop());st.append(ei(min(x,y)))
    elif op==0xa9: st.append(ripemd160(hashlib.sha256(st.pop()).digest()))
    elif op==0xa8: st.append(hashlib.sha256(st.pop()).digest())
    elif op==0xa6: st.append(ripemd160(st.pop()))
    elif op==0x88:
        a=st.pop();c=st.pop()
        if a==c: hp+=1
        elif fail is None: fail=('HORS',)
    elif op==0xad:
        pub=st.pop();sig=st.pop()
        if not csig(sig,pub,[sig]) and fail is None: fail=('CHECKSIG',)
    elif op==0xae:
        nk=di(st.pop()); pubs=[st.pop() for _ in range(nk)][::-1]
        ns=di(st.pop()); sigs=[st.pop() for _ in range(ns)][::-1]; st.pop()
        allcs=[x for x in sigs if parse_der(x)]; si=0; mt=0
        for sg in sigs:
            while si<nk:
                if csig(sg,pubs[si],allcs): mt+=1; si+=1; break
                si+=1
            else: break
        st.append(b'\x01' if mt==ns else b''); cms+=1
        if mt!=ns and fail is None: fail=('CMS',mt,ns)
    peak=max(peak,len(st))
truthy=bool(st) and any(x!=0 for x in st[-1])
print(f"  [mine] HORS {hp}/15  CMS {cms}/2  peak_stack {peak}  TRUE={truthy}  fail={fail}")

# --- DIAGNOSTIC 2: pinning-ONLY relaxed lock through Core (isolates sighash) ---
pin_only = relax(bytes(bt.QSBScriptBuilder.build_pinning_script(b, pin_sig)))
pin_wit = push_data(b'\x02'+b'\x00'*32) + push_data(knp)
core_check(pin_only, pin_wit, "pinning-only")

# --- full relaxed lock through Core ---
print("\nrunning qsb_verify (real libbitcoinconsensus) on the FULL QSB input...\n")
d=tempfile.mkdtemp()
sp=os.path.join(d,'spend.hex'); fp=os.path.join(d,'fund.hex')
open(sp,'w').write(tx.serialize().hex()); open(fp,'w').write(ftx.serialize().hex())
r=subprocess.run([QV, sp, fp, "0", "1"], capture_output=True, text=True)
print(r.stdout[-800:] if r.stdout else r.stderr[-800:])
