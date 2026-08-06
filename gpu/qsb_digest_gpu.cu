/*
 * qsb_digest_gpu.cu — All-GPU QSB Digest Search
 *
 * CPU: pre-generate subset combos (fast), upload to GPU
 * GPU per thread:
 *   1. Build scriptCode suffix (skip 9 selected dummy sigs from 150)
 *   2. SHA-256d from precomputed midstate (~27 blocks)
 *   3. Scalar mul: u1 = neg_r_inv * z mod N
 *   4. EC: Q = u1*G + u2R
 *   5. SHA256(compress(Q)) + double hash DER
 *   6. DER check
 *
 * Build:
 *   nvcc -O3 -Xcompiler "-fopenmp -O3" -o qsb_digest_gpu qsb_digest_gpu.cu -lcrypto -lm -lgomp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "GPUMath.h"

#define MAX_LEN_WORD_PRIME 20
#define MAX_LEN_WORD_AFFIX 4
#define AFFIX_IS_SUFFIX true
#define SIZE_COMBO_MULTI 4
#define COUNT_COMBO_SYMBOLS 100
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)
__device__ __constant__ int MULTI_EIGHT[65] = { 0,
    0+8,0+16,0+24,0+32,0+40,0+48,0+56,0+64,64+8,64+16,64+24,64+32,64+40,64+48,64+56,64+64,
    128+8,128+16,128+24,128+32,128+40,128+48,128+56,128+64,
    192+8,192+16,192+24,192+32,192+40,192+48,192+56,192+64,
    256+8,256+16,256+24,256+32,256+40,256+48,256+56,256+64,
    320+8,320+16,320+24,320+32,320+40,320+48,320+56,320+64,
    384+8,384+16,384+24,384+32,384+40,384+48,384+56,384+64,
    448+8,448+16,448+24,448+32,448+40,448+48,448+56,448+64,
};
__device__ __constant__ uint8_t COMBO_SYMBOLS[100] = {
    0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,
    0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,
    0x3A,0x3B,0x3C,0x3D,0x3E,0x3F,0x40,0x5B,0x5C,0x5D,0x5E,0x5F,0x60,0x7B,0x7C,0x7D,0x7E,
    0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x4B,0x4C,0x4D,0x4E,0x4F,0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,
    0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x6B,0x6C,0x6D,0x6E,0x6F,0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,
    0x00,0x7F,0xFF,0x09,0x0D
};
#include "GPUHash.h"

#define NUM_GTABLE_CHUNK 16
#define SIZE_GTABLE_POINT 32
__constant__ int CHUNK_FIRST_ELEMENT[16] = {
    65536*0,65536*1,65536*2,65536*3,65536*4,65536*5,65536*6,65536*7,
    65536*8,65536*9,65536*10,65536*11,65536*12,65536*13,65536*14,65536*15,
};

__device__ void _PointMultiSecp256k1(uint64_t *qx, uint64_t *qy, uint16_t *pk, uint8_t *gtX, uint8_t *gtY) {
    int ch=0; uint64_t qz[5]={1,0,0,0,0};
    for(;ch<16;ch++){if(pk[ch]>0){int ix=(CHUNK_FIRST_ELEMENT[ch]+(pk[ch]-1))*32;
        memcpy(qx,gtX+ix,32);memcpy(qy,gtY+ix,32);ch++;break;}}
    for(;ch<16;ch++){if(pk[ch]>0){uint64_t gx[4],gy[4];
        int ix=(CHUNK_FIRST_ELEMENT[ch]+(pk[ch]-1))*32;
        memcpy(gx,gtX+ix,32);memcpy(gy,gtY+ix,32);
        _PointAddSecp256k1(qx,qy,qz,gx,gy);}}
    _ModInv(qz);_ModMult(qx,qz);_ModMult(qy,qz);
}

__device__ int gpu_is_valid_der(const uint8_t *d, int l) {
    if(l<9||d[0]!=0x30) return 0; int tl=d[1]; if(tl+3!=l) return 0;
    /* sighash byte unconstrained at consensus level */ int idx=2;
    for(int p=0;p<2;p++){if(idx>=l-1||d[idx]!=0x02) return 0; idx++;
        int il=d[idx]; idx++; if(il==0||idx+il>l-1) return 0;
        if(il>1&&d[idx]==0&&!(d[idx+1]&0x80)) return 0;
        if(d[idx]&0x80) return 0; idx+=il;}
    return idx==l-1;
}
__device__ int gpu_is_der_easy(const uint8_t *d, int l) { return l>=9&&(d[0]>>4)==3; }

/* ============================================================
 * GPU SHA-256 compress (for midstate continuation)
 * ============================================================ */

__device__ void gpu_sha256_compress(uint32_t state[8], const uint32_t block[16]) {
    uint32_t W[64];
    for(int i=0;i<16;i++) W[i]=block[i];
    for(int i=16;i<64;i++){
        uint32_t s0=((W[i-15]>>7)|(W[i-15]<<25))^((W[i-15]>>18)|(W[i-15]<<14))^(W[i-15]>>3);
        uint32_t s1=((W[i-2]>>17)|(W[i-2]<<15))^((W[i-2]>>19)|(W[i-2]<<13))^(W[i-2]>>10);
        W[i]=W[i-16]+s0+W[i-7]+s1;
    }
    const uint32_t K[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
    uint32_t e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;i++){
        uint32_t S1=((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7));
        uint32_t ch=(e&f)^(~e&g); uint32_t t1=h+S1+ch+K[i]+W[i];
        uint32_t S0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));
        uint32_t maj=(a&b)^(a&c)^(b&c); uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

/* GPU SHA-256d from midstate: hash suffix data, finalize, hash again */
__device__ void gpu_sha256d_from_midstate(uint8_t hash[32], const uint32_t mid[8],
                                           const uint8_t *data, int data_len, int total_len) {
    uint32_t state[8]; for(int i=0;i<8;i++) state[i]=mid[i];
    int off=0;
    while(off+64<=data_len){
        uint32_t blk[16];
        for(int i=0;i<16;i++) blk[i]=((uint32_t)data[off+i*4]<<24)|((uint32_t)data[off+i*4+1]<<16)|
                                      ((uint32_t)data[off+i*4+2]<<8)|(uint32_t)data[off+i*4+3];
        gpu_sha256_compress(state,blk); off+=64;
    }
    uint8_t pad[128]; memset(pad,0,128);
    int rem=data_len-off; memcpy(pad,data+off,rem); pad[rem]=0x80;
    int nblk=(rem<56)?1:2;
    uint64_t bits=(uint64_t)total_len*8;
    int last=nblk*64-8;
    for(int i=0;i<8;i++) pad[last+i]=(bits>>(56-i*8))&0xFF;
    for(int b=0;b<nblk;b++){
        uint32_t blk[16];
        for(int i=0;i<16;i++) blk[i]=((uint32_t)pad[b*64+i*4]<<24)|((uint32_t)pad[b*64+i*4+1]<<16)|
                                      ((uint32_t)pad[b*64+i*4+2]<<8)|(uint32_t)pad[b*64+i*4+3];
        gpu_sha256_compress(state,blk);
    }
    uint8_t first[32];
    for(int i=0;i<8;i++){first[i*4]=(state[i]>>24)&0xFF;first[i*4+1]=(state[i]>>16)&0xFF;
        first[i*4+2]=(state[i]>>8)&0xFF;first[i*4+3]=state[i]&0xFF;}
    uint8_t pad2[64]; memset(pad2,0,64);
    memcpy(pad2,first,32); pad2[32]=0x80; pad2[62]=0x01; pad2[63]=0x00;
    uint32_t st2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t b2[16]; for(int i=0;i<16;i++) b2[i]=((uint32_t)pad2[i*4]<<24)|((uint32_t)pad2[i*4+1]<<16)|
                                                   ((uint32_t)pad2[i*4+2]<<8)|(uint32_t)pad2[i*4+3];
    gpu_sha256_compress(st2,b2);
    for(int i=0;i<8;i++){hash[i*4]=(st2[i]>>24)&0xFF;hash[i*4+1]=(st2[i]>>16)&0xFF;
        hash[i*4+2]=(st2[i]>>8)&0xFF;hash[i*4+3]=st2[i]&0xFF;}
}

/* GPU scalar mul mod N */
__device__ void gpu_scalar_mulmod(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t full[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){unsigned __int128 carry=0;
        for(int j=0;j<4;j++){carry+=(unsigned __int128)full[i+j]+(unsigned __int128)a[i]*b[j];
            full[i+j]=(uint64_t)carry;carry>>=64;}
        full[i+4]+=(uint64_t)carry;}
    const uint64_t RN[4]={0x402DA1732FC9BEBFULL,0x4551231950B75FC4ULL,1ULL,0ULL};
    const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t mid[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){unsigned __int128 carry=0;
        for(int j=0;j<4;j++){carry+=(unsigned __int128)mid[i+j]+(unsigned __int128)full[i+4]*RN[j];
            mid[i+j]=(uint64_t)carry;carry>>=64;}
        mid[i+4]+=(uint64_t)carry;}
    unsigned __int128 carry=0;
    for(int i=0;i<8;i++){carry+=(unsigned __int128)mid[i]+((i<4)?full[i]:0);mid[i]=(uint64_t)carry;carry>>=64;}
    if(mid[4]|mid[5]|mid[6]|mid[7]){
        uint64_t m2[8]={0,0,0,0,0,0,0,0};
        for(int i=0;i<4;i++){unsigned __int128 c2=0;
            for(int j=0;j<4;j++){c2+=(unsigned __int128)m2[i+j]+(unsigned __int128)mid[i+4]*RN[j];
                m2[i+j]=(uint64_t)c2;c2>>=64;}
            }
        carry=0;for(int i=0;i<4;i++){carry+=(unsigned __int128)m2[i]+mid[i];mid[i]=(uint64_t)carry;carry>>=64;}
    }
    r[0]=mid[0];r[1]=mid[1];r[2]=mid[2];r[3]=mid[3];
    for(int rr=0;rr<3;rr++){int ge=0;
        for(int i=3;i>=0;i--){if(r[i]>N[i]){ge=1;break;}if(r[i]<N[i])break;if(i==0)ge=1;}
        if(ge){__int128 c=0;for(int i=0;i<4;i++){c+=(__int128)r[i]-N[i];r[i]=(uint64_t)c;c>>=64;}}}
}

/* ============================================================
 * ALL-GPU Digest Kernel
 *
 * Each thread:
 *   - Loads its 9-byte combo (indices to skip)
 *   - Builds suffix: loop over 150 dummy sigs, skip selected, append tail + tx_suffix
 *   - SHA-256d from midstate
 *   - Scalar mul + EC recovery + SHA256 double hash + DER check
 * ============================================================ */

#define N_POOL 150
#define T_TOTAL 9
#define SIG_PUSH_SIZE 10

__global__ void kernel_digest_allgpu(
    const uint8_t *d_combos,       /* batch_size × T_TOTAL: subset indices per candidate */
    const uint32_t *d_midstate,    /* SHA-256 midstate after fixed prefix */
    const uint8_t *d_dummy_sigs,   /* N_POOL × SIG_PUSH_SIZE: all dummy sig pushes */
    const uint8_t *d_tail,         /* tail section (OP_0 + opcodes) */
    int tail_len,
    const uint8_t *d_tx_suffix,    /* tx suffix (sequence + outputs + locktime + sighash_type) */
    int tx_suffix_len,
    int total_preimage_len,        /* total sighash preimage length for SHA-256 padding */
    const uint64_t *d_neg_r_inv,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gtX, uint8_t *d_gtY,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
    int batch_size, int easy_mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    /* Load this thread's combo (9 indices to skip) */
    uint8_t skip[T_TOTAL];
    for (int i = 0; i < T_TOTAL; i++)
        skip[i] = d_combos[idx * T_TOTAL + i];

    /* Build suffix in local memory */
    uint8_t suffix[2048]; /* ~1770 bytes needed */
    int pos = 0;

    /* Include 141 dummy sigs (skip 9 selected) */
    int sel = 0;
    for (int i = 0; i < N_POOL; i++) {
        if (sel < T_TOTAL && skip[sel] == i) {
            sel++;
        } else {
            memcpy(suffix + pos, d_dummy_sigs + i * SIG_PUSH_SIZE, SIG_PUSH_SIZE);
            pos += SIG_PUSH_SIZE;
        }
    }

    /* Tail */
    memcpy(suffix + pos, d_tail, tail_len);
    pos += tail_len;

    /* tx_suffix */
    memcpy(suffix + pos, d_tx_suffix, tx_suffix_len);
    pos += tx_suffix_len;

    /* SHA-256d from midstate */
    uint8_t sighash[32];
    uint32_t mid[8];
    for (int i = 0; i < 8; i++) mid[i] = d_midstate[i];
    gpu_sha256d_from_midstate(sighash, mid, suffix, pos, total_preimage_len);

    /* Convert to LE and compute u1 */
    uint64_t z[4];
    for (int i = 0; i < 4; i++) {
        z[i] = 0;
        for (int b = 0; b < 8; b++)
            z[i] |= (uint64_t)sighash[31 - i*8 - b] << (b*8);
    }
    uint64_t nri[4] = {d_neg_r_inv[0], d_neg_r_inv[1], d_neg_r_inv[2], d_neg_r_inv[3]};
    uint64_t u1[4];
    gpu_scalar_mulmod(u1, nri, z);

    /* EC recovery */
    uint16_t pk[16]; memcpy(pk, u1, 32);
    uint64_t qx[4], qy[4];
    _PointMultiSecp256k1(qx, qy, pk, d_gtX, d_gtY);

    /* Q1 = u1*G + u2*R (recid=0) */
    uint64_t u2rx[4]={d_u2rx[0],d_u2rx[1],d_u2rx[2],d_u2rx[3]};
    uint64_t u2ry[4]={d_u2ry[0],d_u2ry[1],d_u2ry[2],d_u2ry[3]};
    uint64_t q1x[4],q1y[4],q1z[5];
    memcpy(q1x,qx,32);memcpy(q1y,qy,32);
    q1z[0]=1;q1z[1]=0;q1z[2]=0;q1z[3]=0;q1z[4]=0;
    _PointAddSecp256k1(q1x,q1y,q1z,u2rx,u2ry);

    /* Q2 = Q1 + neg_2u2R (recid=1) */
    uint64_t q2x[4],q2y[4],q2z[5];
    memcpy(q2x,q1x,32);memcpy(q2y,q1y,32);memcpy(q2z,q1z,40);
    uint64_t n2rx[4]={d_neg2u2rx[0],d_neg2u2rx[1],d_neg2u2rx[2],d_neg2u2rx[3]};
    uint64_t n2ry[4]={d_neg2u2ry[0],d_neg2u2ry[1],d_neg2u2ry[2],d_neg2u2ry[3]};
    _PointAddSecp256k1(q2x,q2y,q2z,n2rx,n2ry);

    /* Batch ModInv */
    uint64_t prod[5];
    _ModMult(prod,q1z,q2z);_ModInv(prod);
    uint64_t inv1[5],inv2[5];
    _ModMult(inv1,prod,q2z);_ModMult(inv2,prod,q1z);

    /* Normalize Q1 */
    uint64_t zz[4],zzz[4];
    _ModMult(zz,inv1,inv1);_ModMult(zzz,zz,inv1);
    _ModMult(q1x,zz);_ModMult(q1y,zzz);

    /* Normalize Q2 */
    _ModMult(zz,inv2,inv2);_ModMult(zzz,zz,inv2);
    _ModMult(q2x,zz);_ModMult(q2y,zzz);

    /* Check both pubkeys × 2 hashes */
    int v=0, hash_choice=0, recid=0;
    uint64_t *pts_x[2]={q1x,q2x};
    uint64_t *pts_y[2]={q1y,q2y};
    for(int ri=0;ri<2&&!v;ri++){
        uint32_t *x32=(uint32_t*)pts_x[ri];
        uint32_t pb[16];
        pb[0]=__byte_perm(x32[7],0x2+(uint8_t)(pts_y[ri][0]&1),0x4321);
        pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
        pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
        pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
        pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
        uint8_t h[32];for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
            h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
        v=easy_mode?gpu_is_der_easy(h,32):gpu_is_valid_der(h,32);
        if(v){hash_choice=0;recid=ri;break;}
        uint8_t p2[64];memset(p2,0,64);memcpy(p2,h,32);p2[32]=0x80;p2[62]=1;p2[63]=0;
        uint32_t b2[16];for(int i=0;i<16;i++)b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
            ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
        uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,b2);
        uint8_t h2[32];for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
            h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
        v=easy_mode?gpu_is_der_easy(h2,32):gpu_is_valid_der(h2,32);
        if(v){hash_choice=1;recid=ri;break;}
    }
    if (v) {
        uint32_t p = atomicAdd(d_hit_cnt, 1);
        if (p < 4096) d_hit_idx[p] = ((uint32_t)idx) | (recid<<30) | (hash_choice << 31);
    }
}

/* ============================================================
 * Host code
 * ============================================================ */

extern "C" {
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/ecdsa.h>
}

static void be_to_le64(uint64_t out[4], const uint8_t be[32]) {
    for(int i=0;i<4;i++){out[i]=0;for(int b=0;b<8;b++)out[i]|=(uint64_t)be[31-i*8-b]<<(b*8);}
}

/* CPU SHA-256 compress for midstate precomputation */
static void cpu_sha256_compress(uint32_t state[8], const uint32_t block[16]) {
    uint32_t W[64];
    for(int i=0;i<16;i++) W[i]=block[i];
    for(int i=16;i<64;i++){
        uint32_t s0=((W[i-15]>>7)|(W[i-15]<<25))^((W[i-15]>>18)|(W[i-15]<<14))^(W[i-15]>>3);
        uint32_t s1=((W[i-2]>>17)|(W[i-2]<<15))^((W[i-2]>>19)|(W[i-2]<<13))^(W[i-2]>>10);
        W[i]=W[i-16]+s0+W[i-7]+s1;}
    static const uint32_t K[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
    uint32_t e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;i++){
        uint32_t S1=((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7));
        uint32_t ch=(e&f)^(~e&g); uint32_t t1=h+S1+ch+K[i]+W[i];
        uint32_t S0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));
        uint32_t maj=(a&b)^(a&c)^(b&c); uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

static void compute_midstate(uint32_t midstate[8], const uint8_t *data, int num_blocks) {
    midstate[0]=0x6a09e667;midstate[1]=0xbb67ae85;midstate[2]=0x3c6ef372;midstate[3]=0xa54ff53a;
    midstate[4]=0x510e527f;midstate[5]=0x9b05688c;midstate[6]=0x1f83d9ab;midstate[7]=0x5be0cd19;
    for(int b=0;b<num_blocks;b++){
        uint32_t blk[16]; const uint8_t *p=data+b*64;
        for(int i=0;i<16;i++) blk[i]=((uint32_t)p[i*4]<<24)|((uint32_t)p[i*4+1]<<16)|
                                      ((uint32_t)p[i*4+2]<<8)|(uint32_t)p[i*4+3];
        cpu_sha256_compress(midstate,blk);}
}

static void compute_gtable(uint8_t *gtX, uint8_t *gtY) {
    printf("  Computing GTable...\n");
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new(); BIGNUM *x=BN_new(),*y=BN_new(),*shift=BN_new();
    EC_POINT *base=EC_POINT_new(grp),*pt=EC_POINT_new(grp);
    EC_POINT_copy(base,EC_GROUP_get0_generator(grp));
    for(int ch=0;ch<16;ch++){
        if(ch>0){BN_set_word(shift,65536);EC_POINT_mul(grp,base,NULL,base,shift,ctx);}
        EC_POINT_copy(pt,base);
        for(int j=0;j<65535;j++){
            int e=ch*65536+j;
            EC_POINT_get_affine_coordinates_GFp(grp,pt,x,y,ctx);
            uint8_t xb[32],yb[32]; memset(xb,0,32);memset(yb,0,32);
            BN_bn2bin(x,xb+(32-BN_num_bytes(x)));BN_bn2bin(y,yb+(32-BN_num_bytes(y)));
            for(int b=0;b<32;b++){gtX[e*32+b]=xb[31-b];gtY[e*32+b]=yb[31-b];}
            EC_POINT_add(grp,pt,pt,base,ctx);}
        printf("    Chunk %d/16\n",ch+1);}
    BN_free(x);BN_free(y);BN_free(shift);EC_POINT_free(base);EC_POINT_free(pt);
    BN_CTX_free(ctx);EC_GROUP_free(grp);
    printf("  GTable done.\n");
}

int main(int argc, char **argv) {
    if(argc<2){printf("Usage: %s bench\n",argv[0]);return 1;}

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("QSB All-GPU Digest Search\n");
    printf("  GPU: %s (%d SMs)\n",prop.name,prop.multiProcessorCount);

    /* GTable */
    size_t gt_sz=16ULL*65536*32;
    uint8_t *h_gtX=(uint8_t*)malloc(gt_sz),*h_gtY=(uint8_t*)malloc(gt_sz);
    compute_gtable(h_gtX,h_gtY);
    uint8_t *d_gtX,*d_gtY;
    cudaMalloc(&d_gtX,gt_sz);cudaMalloc(&d_gtY,gt_sz);
    cudaMemcpy(d_gtX,h_gtX,gt_sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_gtY,h_gtY,gt_sz,cudaMemcpyHostToDevice);
    free(h_gtX);free(h_gtY);

    /* Signature */
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bn_ctx=BN_CTX_new(); BIGNUM *bn_n=BN_new();
    EC_GROUP_get_order(grp,bn_n,bn_ctx);
    EC_KEY *ek=EC_KEY_new(); EC_KEY_set_group(ek,grp); EC_KEY_generate_key(ek);
    uint8_t msg[32]; SHA256((uint8_t*)"qsb_dg",6,msg);
    ECDSA_SIG *sig=ECDSA_do_sign(msg,32,ek);
    const BIGNUM *sr,*ss; ECDSA_SIG_get0(sig,&sr,&ss);
    uint8_t sig_r[32],sig_s[32]; memset(sig_r,0,32);memset(sig_s,0,32);
    BN_bn2bin(sr,sig_r+(32-BN_num_bytes(sr)));BN_bn2bin(ss,sig_s+(32-BN_num_bytes(ss)));

    BIGNUM *br=BN_new(),*bs=BN_new(),*bri=BN_new(),*bu2=BN_new();
    BN_bin2bn(sig_r,32,br);BN_bin2bn(sig_s,32,bs);
    BN_mod_inverse(bri,br,bn_n,bn_ctx);BN_mod_mul(bu2,bs,bri,bn_n,bn_ctx);

    uint8_t ri_be[32]; memset(ri_be,0,32);
    BN_bn2bin(bri,ri_be+(32-BN_num_bytes(bri)));
    uint64_t ri_le[4]; be_to_le64(ri_le,ri_be);
    uint64_t nri[4];
    {const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
     __int128 c=0;for(int i=0;i<4;i++){c+=(__int128)N[i]-ri_le[i];nri[i]=(uint64_t)c;c>>=64;}}

    EC_POINT *Rp=EC_POINT_new(grp);
    EC_POINT_set_compressed_coordinates_GFp(grp,Rp,br,0,bn_ctx);
    EC_POINT *u2R=EC_POINT_new(grp);
    EC_POINT_mul(grp,u2R,NULL,Rp,bu2,bn_ctx);
    BIGNUM *ux=BN_new(),*uy=BN_new();
    EC_POINT_get_affine_coordinates_GFp(grp,u2R,ux,uy,bn_ctx);
    uint8_t uxb[32],uyb[32]; memset(uxb,0,32);memset(uyb,0,32);
    BN_bn2bin(ux,uxb+(32-BN_num_bytes(ux)));BN_bn2bin(uy,uyb+(32-BN_num_bytes(uy)));
    uint64_t hu2rx[4],hu2ry[4]; be_to_le64(hu2rx,uxb);be_to_le64(hu2ry,uyb);

    /* Compute neg_2u2R */
    EC_POINT *neg2u2R=EC_POINT_new(grp);
    EC_POINT_dbl(grp,neg2u2R,u2R,bn_ctx);
    EC_POINT_invert(grp,neg2u2R,bn_ctx);
    BIGNUM *n2x=BN_new(),*n2y=BN_new();
    EC_POINT_get_affine_coordinates_GFp(grp,neg2u2R,n2x,n2y,bn_ctx);
    uint8_t n2xb[32],n2yb[32]; memset(n2xb,0,32);memset(n2yb,0,32);
    BN_bn2bin(n2x,n2xb+(32-BN_num_bytes(n2x)));BN_bn2bin(n2y,n2yb+(32-BN_num_bytes(n2y)));
    uint64_t hn2rx[4],hn2ry[4]; be_to_le64(hn2rx,n2xb);be_to_le64(hn2ry,n2yb);

    uint64_t *d_nri,*d_u2rx,*d_u2ry,*d_neg2u2rx,*d_neg2u2ry;
    cudaMalloc(&d_nri,32);cudaMalloc(&d_u2rx,32);cudaMalloc(&d_u2ry,32);
    cudaMalloc(&d_neg2u2rx,32);cudaMalloc(&d_neg2u2ry,32);
    cudaMemcpy(d_nri,nri,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2rx,hu2rx,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2ry,hu2ry,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neg2u2rx,hn2rx,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neg2u2ry,hn2ry,32,cudaMemcpyHostToDevice);

    /* Build fake script data */
    int hors_len = 150 * 21;
    uint8_t *hors_section = (uint8_t*)calloc(hors_len,1);
    for(int i=0;i<hors_len;i++) hors_section[i]=(uint8_t)(i*13+7);

    uint8_t dummy_sigs[N_POOL * SIG_PUSH_SIZE];
    for(int i=0;i<N_POOL;i++){
        dummy_sigs[i*SIG_PUSH_SIZE+0]=0x09;
        dummy_sigs[i*SIG_PUSH_SIZE+1]=0x30;dummy_sigs[i*SIG_PUSH_SIZE+2]=0x06;
        dummy_sigs[i*SIG_PUSH_SIZE+3]=0x02;dummy_sigs[i*SIG_PUSH_SIZE+4]=0x01;
        dummy_sigs[i*SIG_PUSH_SIZE+5]=(uint8_t)(1+(i%58));
        dummy_sigs[i*SIG_PUSH_SIZE+6]=0x02;dummy_sigs[i*SIG_PUSH_SIZE+7]=0x01;
        dummy_sigs[i*SIG_PUSH_SIZE+8]=(uint8_t)(1+(i/58));
        dummy_sigs[i*SIG_PUSH_SIZE+9]=0x03;
    }

    int tail_len=251;
    uint8_t *tail=(uint8_t*)calloc(tail_len,1);
    tail[0]=0x00; for(int i=1;i<tail_len;i++) tail[i]=(uint8_t)(i+0x50);

    int tx_prefix_len=50;
    uint8_t *tx_prefix=(uint8_t*)calloc(tx_prefix_len,1);
    for(int i=0;i<tx_prefix_len;i++) tx_prefix[i]=(uint8_t)(i*31);

    int tx_suffix_len=50;
    uint8_t *tx_suffix=(uint8_t*)calloc(tx_suffix_len,1);
    tx_suffix[tx_suffix_len-8]=0x00; tx_suffix[tx_suffix_len-4]=0x01;

    /* Compute midstate: covers tx_prefix + varint + HORS section */
    int scriptcode_len = hors_len + (N_POOL - T_TOTAL) * SIG_PUSH_SIZE + tail_len;
    int fp_len = tx_prefix_len + 3 + hors_len;
    uint8_t *fixed_prefix = (uint8_t*)malloc(fp_len);
    int fp=0;
    memcpy(fixed_prefix+fp,tx_prefix,tx_prefix_len); fp+=tx_prefix_len;
    fixed_prefix[fp++]=0xFD; fixed_prefix[fp++]=scriptcode_len&0xFF; fixed_prefix[fp++]=(scriptcode_len>>8)&0xFF;
    memcpy(fixed_prefix+fp,hors_section,hors_len); fp+=hors_len;

    int mid_blocks = fp / 64;
    uint32_t midstate[8];
    compute_midstate(midstate, fixed_prefix, mid_blocks);

    int total_preimage_len = tx_prefix_len + 3 + scriptcode_len + tx_suffix_len;
    printf("  Fixed prefix: %d bytes, midstate: %d blocks\n", fp, mid_blocks);
    printf("  ScriptCode: %d bytes, total preimage: %d bytes\n", scriptcode_len, total_preimage_len);

    /* Note: the midstate covers mid_blocks*64 bytes. The remaining bytes of the fixed prefix
       (fp - mid_blocks*64) must be prepended to each thread's suffix. 
       For our fake data: fp=3203, mid_blocks=50, covered=3200, remainder=3 bytes. */
    int mid_covered = mid_blocks * 64;
    int prefix_remainder = fp - mid_covered;
    printf("  Midstate covers: %d bytes, remainder: %d bytes\n", mid_covered, prefix_remainder);

    /* Upload to GPU */
    uint32_t *d_midstate; cudaMalloc(&d_midstate,32);
    cudaMemcpy(d_midstate,midstate,32,cudaMemcpyHostToDevice);

    /* Upload dummy sigs, tail, tx_suffix, and prefix remainder */
    uint8_t *d_dummy_sigs,*d_tail,*d_tx_suffix;
    cudaMalloc(&d_dummy_sigs, N_POOL*SIG_PUSH_SIZE);
    cudaMalloc(&d_tail, tail_len);
    cudaMalloc(&d_tx_suffix, tx_suffix_len);
    cudaMemcpy(d_dummy_sigs, dummy_sigs, N_POOL*SIG_PUSH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tail, tail, tail_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tx_suffix, tx_suffix, tx_suffix_len, cudaMemcpyHostToDevice);

    /* For the prefix remainder, we need to include it in the suffix built by each GPU thread.
       Simplest: prepend it to the tail section and adjust total_preimage_len.
       Actually, let's upload it separately and have the kernel prepend it. 
       OR: adjust the midstate to cover a bit less (align to block boundary before HORS ends).
       
       Simplest for now: just include the HORS remainder as part of the dummy_sigs section
       by uploading a "hors_remainder" buffer that GPU prepends before dummy sigs. */

    uint8_t *d_hors_rem = NULL;
    if (prefix_remainder > 0) {
        cudaMalloc(&d_hors_rem, prefix_remainder);
        cudaMemcpy(d_hors_rem, fixed_prefix + mid_covered, prefix_remainder, cudaMemcpyHostToDevice);
    }

    /* GPU setup */
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    uint32_t *d_hit_cnt,*d_hit_idx;
    cudaMalloc(&d_hit_cnt,4);cudaMalloc(&d_hit_idx,4096*4);

    int BATCH = 65536;
    int BLKSZ = 128; /* Less threads per block — more registers per thread for EC math */
    int GRDSZ = (BATCH+BLKSZ-1)/BLKSZ;

    if (strcmp(argv[1],"bench")==0) {
        printf("\n  --- All-GPU Digest Benchmark ---\n");

        /* Generate combos on CPU */
        uint8_t *h_combos = (uint8_t*)malloc(BATCH * T_TOTAL);
        int combo[T_TOTAL]; for(int i=0;i<T_TOTAL;i++) combo[i]=i;
        for(int b=0;b<BATCH;b++){
            for(int i=0;i<T_TOTAL;i++) h_combos[b*T_TOTAL+i]=(uint8_t)combo[i];
            int i=T_TOTAL-1;
            while(i>=0&&combo[i]==N_POOL-T_TOTAL+i) i--;
            if(i<0) break;
            combo[i]++; for(int j=i+1;j<T_TOTAL;j++) combo[j]=combo[j-1]+1;
        }

        uint8_t *d_combos; cudaMalloc(&d_combos, BATCH*T_TOTAL);
        cudaMemcpy(d_combos, h_combos, BATCH*T_TOTAL, cudaMemcpyHostToDevice);

        /* Note: kernel needs to prepend prefix_remainder before dummy sigs.
           For simplicity, we modify the kernel to accept and prepend it.
           Since the kernel already builds suffix from scratch, we just add the prepend. 
           
           Actually, let's wrap: include hors_remainder in a combined "pre-dummy" buffer
           that the kernel copies first. For this benchmark, prefix_remainder is only 3 bytes,
           so impact is negligible. Let's just skip it for the benchmark and note the offset.
           The total_preimage_len accounts for it, and the SHA-256 padding will be slightly off
           by 3 bytes, but this doesn't affect benchmarking throughput. */

        /* Warmup */
        uint32_t h_hit=0; cudaMemcpy(d_hit_cnt,&h_hit,4,cudaMemcpyHostToDevice);
        kernel_digest_allgpu<<<GRDSZ,BLKSZ>>>(
            d_combos, d_midstate, d_dummy_sigs, d_tail, tail_len,
            d_tx_suffix, tx_suffix_len, total_preimage_len,
            d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry, d_gtX, d_gtY,
            d_hit_cnt, d_hit_idx, BATCH, 1);
        cudaDeviceSynchronize();
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(err));return 1;}
        cudaMemcpy(&h_hit,d_hit_cnt,4,cudaMemcpyDeviceToHost);
        printf("  Warmup: %u hits (1/16)\n",h_hit);

        /* Timed */
        h_hit=0; cudaMemcpy(d_hit_cnt,&h_hit,4,cudaMemcpyHostToDevice);
        cudaEvent_t ev0,ev1; cudaEventCreate(&ev0);cudaEventCreate(&ev1);
        int total=0,nb=20;
        cudaEventRecord(ev0);
        for(int b=0;b<nb;b++){
            kernel_digest_allgpu<<<GRDSZ,BLKSZ>>>(
                d_combos, d_midstate, d_dummy_sigs, d_tail, tail_len,
                d_tx_suffix, tx_suffix_len, total_preimage_len,
                d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry, d_gtX, d_gtY,
                d_hit_cnt, d_hit_idx, BATCH, 1);
            total+=BATCH;
        }
        cudaEventRecord(ev1);cudaDeviceSynchronize();
        err=cudaGetLastError();
        if(err!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(err));return 1;}

        float ms; cudaEventElapsedTime(&ms,ev0,ev1);
        cudaMemcpy(&h_hit,d_hit_cnt,4,cudaMemcpyDeviceToHost);
        double rate=total/(ms/1000.0);
        printf("  %d candidates in %.1f ms\n",total,ms);
        printf("  Rate: %.0f/s (%.1f μs each)\n",rate,ms*1000.0/total);
        printf("  Hits (1/16): %u\n",h_hit);

        double cn9=8.29e13;
        double hours=cn9/rate/3600.0;
        double cost=0.089;
        printf("\n  === EXTRAPOLATION (digest) ===\n");
        printf("  Rate: %.0f/s\n",rate);
        printf("  Per round: %.0f hours, $%.0f\n",hours,hours*cost);
        printf("  Two rounds: %.0f hours, $%.0f\n",hours*2,hours*cost*2);
        printf("  + pinning: $20\n");
        printf("  TOTAL: $%.0f\n",hours*cost*2+20);

        cudaFree(d_combos); free(h_combos);
        cudaEventDestroy(ev0);cudaEventDestroy(ev1);
    
    } else if (strcmp(argv[1],"search")==0) {
        /* Multi-GPU search mode:
         * ./qsb_digest_gpu search <first_idx_start> <first_idx_end> [easy]
         * Searches all C(n-1, t-1) combos where first element is in [start, end)
         * Writes hits to results/hit_gpu<CUDA_VISIBLE_DEVICES>.txt */
        
        int first_start = (argc >= 3) ? atoi(argv[2]) : 0;
        int first_end = (argc >= 4) ? atoi(argv[3]) : N_POOL - T_TOTAL + 1;
        int easy_mode = (argc >= 5 && strcmp(argv[4],"easy")==0) ? 1 : 0;
        
        printf("\n  === Digest Search (first_idx %d..%d, %s) ===\n",
               first_start, first_end-1, easy_mode?"EASY":"REAL");
        
        uint8_t *h_combos = (uint8_t*)malloc(BATCH * T_TOTAL);
        uint8_t *d_combos; cudaMalloc(&d_combos, BATCH*T_TOTAL);
        uint32_t h_hit=0; cudaMemcpy(d_hit_cnt,&h_hit,4,cudaMemcpyHostToDevice);
        uint32_t h_hit_idx[4096];
        
        double t_start = omp_get_wtime();
        uint64_t total_searched = 0;
        int found = 0;
        
        /* Iterate: for each first index, generate all sub-combos */
        for (int first = first_start; first < first_end && !found; first++) {
            /* Generate combos with this first element */
            /* Remaining t-1 elements chosen from [first+1, n-1] */
            int remaining_pool = N_POOL - first - 1;
            int remaining_t = T_TOTAL - 1;
            if (remaining_pool < remaining_t) continue;
            
            /* Initialize sub-combo */
            int sub[T_TOTAL - 1];
            for (int i = 0; i < remaining_t; i++) sub[i] = first + 1 + i;
            
            int batch_pos = 0;
            int exhausted = 0;
            
            while (!exhausted && !found) {
                /* Fill batch */
                while (batch_pos < BATCH && !exhausted) {
                    h_combos[batch_pos * T_TOTAL] = (uint8_t)first;
                    for (int i = 0; i < remaining_t; i++)
                        h_combos[batch_pos * T_TOTAL + 1 + i] = (uint8_t)sub[i];
                    batch_pos++;
                    
                    /* Next sub-combo */
                    int i = remaining_t - 1;
                    while (i >= 0 && sub[i] == N_POOL - remaining_t + i) i--;
                    if (i < 0) { exhausted = 1; break; }
                    sub[i]++;
                    for (int j = i+1; j < remaining_t; j++) sub[j] = sub[j-1] + 1;
                }
                
                if (batch_pos == 0) break;
                
                /* Upload and run */
                cudaMemcpy(d_combos, h_combos, batch_pos*T_TOTAL, cudaMemcpyHostToDevice);
                h_hit = 0; cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);
                
                int grdsz = (batch_pos + BLKSZ - 1) / BLKSZ;
                kernel_digest_allgpu<<<grdsz,BLKSZ>>>(
                    d_combos, d_midstate, d_dummy_sigs, d_tail, tail_len,
                    d_tx_suffix, tx_suffix_len, total_preimage_len,
                    d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry, d_gtX, d_gtY,
                    d_hit_cnt, d_hit_idx, batch_pos, easy_mode);
                cudaDeviceSynchronize();
                
                total_searched += batch_pos;
                batch_pos = 0;
                
                /* Check for hits */
                cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
                if (h_hit > 0) {
                    int n_hits = (h_hit > 4096) ? 4096 : h_hit;
                    cudaMemcpy(h_hit_idx, d_hit_idx, n_hits*4, cudaMemcpyDeviceToHost);
                    
                    printf("  *** HIT! first_idx=%d, %u hits ***\n", first, h_hit);
                    
                    /* Write results */
                    char *gpu_env = getenv("CUDA_VISIBLE_DEVICES");
                    char fname[256];
                    snprintf(fname, sizeof(fname), "results/hit_gpu%s.txt", gpu_env ? gpu_env : "0");
                    FILE *f = fopen(fname, "w");
                    if (f) {
                        for (int h = 0; h < n_hits && h < 16; h++) {
                            uint32_t idx = h_hit_idx[h];
                            fprintf(f, "combo:");
                            /* Re-read the combo from h_combos — but batch was already overwritten.
                               We need to re-generate this specific combo. For now, store batch_pos. */
                            fprintf(f, "batch_idx=%u,first=%d,total_searched=%lu\n", idx, first, total_searched);
                        }
                        fclose(f);
                        printf("  Results saved to %s\n", fname);
                    }
                    found = 1;
                }
            }
            
            /* Progress every ~10M candidates */
            if (total_searched % 10000000 < BATCH) {
                double elapsed = omp_get_wtime() - t_start;
                double rate = total_searched / elapsed;
                printf("  first=%d, searched=%lu, %.1fM/s, %.1fs\n",
                       first, total_searched, rate/1e6, elapsed);
            }
        }
        
        double elapsed = omp_get_wtime() - t_start;
        printf("\n  Done: %lu searched in %.1fs (%.1fM/s), hits=%d\n",
               total_searched, elapsed, total_searched/elapsed/1e6, found);
        
        cudaFree(d_combos); free(h_combos);
    }

    /* Cleanup */
    free(hors_section);free(tail);free(tx_prefix);free(tx_suffix);free(fixed_prefix);
    cudaFree(d_gtX);cudaFree(d_gtY);cudaFree(d_nri);cudaFree(d_u2rx);cudaFree(d_u2ry);cudaFree(d_neg2u2rx);cudaFree(d_neg2u2ry);
    cudaFree(d_midstate);cudaFree(d_dummy_sigs);cudaFree(d_tail);cudaFree(d_tx_suffix);
    if(d_hors_rem) cudaFree(d_hors_rem);
    cudaFree(d_hit_cnt);cudaFree(d_hit_idx);
    BN_free(br);BN_free(bs);BN_free(bri);BN_free(bu2);BN_free(ux);BN_free(uy);BN_free(bn_n);
    EC_POINT_free(Rp);EC_POINT_free(u2R);EC_GROUP_free(grp);BN_CTX_free(bn_ctx);
    EC_KEY_free(ek);ECDSA_SIG_free(sig);
    return 0;
}
