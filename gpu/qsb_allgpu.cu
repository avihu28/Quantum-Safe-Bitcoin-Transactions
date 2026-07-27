/*
 * qsb_allgpu.cu — All-GPU QSB Pinning Search with SHA-256 midstate trick
 *
 * SHA-256 double hash puzzle: checks both SHA-256(key) and SHA-256(SHA-256(key))
 * per candidate, doubling the DER probability for ~0.8% extra cost.
 *
 * Key optimization: CPU precomputes SHA-256 state after 78 blocks (4992 bytes).
 * GPU only finalizes the last block (16 bytes: tail + locktime + sighash_type)
 * + second SHA-256 (32 bytes). Total: 2 SHA-256 compresses per thread.
 *
 * Full GPU pipeline per thread:
 *   1. Finalize SHA-256 (1 block) + second SHA-256 (1 block) = SHA-256d
 *   2. Scalar mul: u1 = neg_r_inv * z mod N
 *   3. EC: Q1 = u1*G + u2R, Q2 = Q1 + neg_2u2R (both recovery flags)
 *   4. Batch ModInv (Montgomery trick on Q1.z, Q2.z)
 *   5. SHA-256(compress(Qi)) → DER check × 2 pubkeys × 2 hashes = 4 checks
 *
 * Build:
 *   nvcc -O3 -o qsb_allgpu qsb_allgpu.cu -lcrypto -lm
 *
 * Usage:
 *   ./qsb_allgpu bench
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

/* CudaBrainSecp GPU includes */
#include "GPUMath.h"

#define MAX_LEN_WORD_PRIME 20
#define MAX_LEN_WORD_AFFIX 4
#define AFFIX_IS_SUFFIX true
#define SIZE_COMBO_MULTI 4
#define COUNT_COMBO_SYMBOLS 100
#define IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)

__device__ __constant__ int MULTI_EIGHT[65] = { 0,
    0+8,0+16,0+24,0+32,0+40,0+48,0+56,0+64,
    64+8,64+16,64+24,64+32,64+40,64+48,64+56,64+64,
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

/* GTable */
#define NUM_GTABLE_CHUNK 16
#define NUM_GTABLE_VALUE 65536
#define SIZE_GTABLE_POINT 32

__constant__ int CHUNK_FIRST_ELEMENT[16] = {
    65536*0,65536*1,65536*2,65536*3,65536*4,65536*5,65536*6,65536*7,
    65536*8,65536*9,65536*10,65536*11,65536*12,65536*13,65536*14,65536*15,
};

__device__ void _PointMultiSecp256k1(uint64_t *qx, uint64_t *qy, uint16_t *privKey, uint8_t *gTableX, uint8_t *gTableY) {
    int chunk=0; uint64_t qz[5]={1,0,0,0,0};
    for(;chunk<16;chunk++){if(privKey[chunk]>0){
        int index=(CHUNK_FIRST_ELEMENT[chunk]+(privKey[chunk]-1))*32;
        memcpy(qx,gTableX+index,32);memcpy(qy,gTableY+index,32);chunk++;break;}}
    for(;chunk<16;chunk++){if(privKey[chunk]>0){
        uint64_t gx[4],gy[4];
        int index=(CHUNK_FIRST_ELEMENT[chunk]+(privKey[chunk]-1))*32;
        memcpy(gx,gTableX+index,32);memcpy(gy,gTableY+index,32);
        _PointAddSecp256k1(qx,qy,qz,gx,gy);}}
    _ModInv(qz);_ModMult(qx,qz);_ModMult(qy,qz);
}

/* DER checks */
__device__ int gpu_is_valid_der(const uint8_t *d, int l) {
    if(l<9||d[0]!=0x30) return 0;
    int tl=d[1]; if(tl+3!=l) return 0;
    /* sighash byte unconstrained at consensus level */
    int idx=2;
    for(int p=0;p<2;p++){
        if(idx>=l-1||d[idx]!=0x02) return 0; idx++;
        int il=d[idx]; idx++;
        if(il==0||idx+il>l-1) return 0;
        if(il>1&&d[idx]==0&&!(d[idx+1]&0x80)) return 0;
        if(d[idx]&0x80) return 0; idx+=il;}
    return idx==l-1;
}
__device__ int gpu_is_der_easy(const uint8_t *d, int l) { return l>=9&&(d[0]>>4)==3; }

/* ============================================================
 * GPU SHA-256 compress (single block)
 * ============================================================ */

__device__ void gpu_sha256_compress_block(uint32_t state[8], const uint32_t block[16]) {
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
        uint32_t ch=(e&f)^(~e&g);
        uint32_t t1=h+S1+ch+K[i]+W[i];
        uint32_t S0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

/* ============================================================
 * GPU scalar mul mod N
 * ============================================================ */

__device__ void gpu_scalar_mulmod(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t full[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){
        unsigned __int128 carry=0;
        for(int j=0;j<4;j++){
            carry+=(unsigned __int128)full[i+j]+(unsigned __int128)a[i]*b[j];
            full[i+j]=(uint64_t)carry; carry>>=64;
        }
        full[i+4]+=(uint64_t)carry;
    }
    const uint64_t RN[4]={0x402DA1732FC9BEBFULL,0x4551231950B75FC4ULL,1ULL,0ULL};
    const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t mid[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){
        unsigned __int128 carry=0;
        for(int j=0;j<4;j++){
            carry+=(unsigned __int128)mid[i+j]+(unsigned __int128)full[i+4]*RN[j];
            mid[i+j]=(uint64_t)carry; carry>>=64;
        }
        mid[i+4]+=(uint64_t)carry;
    }
    unsigned __int128 carry=0;
    for(int i=0;i<8;i++){
        carry+=(unsigned __int128)mid[i]+((i<4)?full[i]:0);
        mid[i]=(uint64_t)carry; carry>>=64;
    }
    if(mid[4]|mid[5]|mid[6]|mid[7]){
        uint64_t m2[8]={0,0,0,0,0,0,0,0};
        for(int i=0;i<4;i++){
            unsigned __int128 c2=0;
            for(int j=0;j<4;j++){
                c2+=(unsigned __int128)m2[i+j]+(unsigned __int128)mid[i+4]*RN[j];
                m2[i+j]=(uint64_t)c2; c2>>=64;
            }
        }
        carry=0;
        for(int i=0;i<4;i++){
            carry+=(unsigned __int128)m2[i]+mid[i];
            mid[i]=(uint64_t)carry; carry>>=64;
        }
    }
    r[0]=mid[0];r[1]=mid[1];r[2]=mid[2];r[3]=mid[3];
    for(int rr=0;rr<3;rr++){
        int ge=0;
        for(int i=3;i>=0;i--){if(r[i]>N[i]){ge=1;break;}if(r[i]<N[i])break;if(i==0)ge=1;}
        if(ge){__int128 c=0;for(int i=0;i<4;i++){c+=(__int128)r[i]-N[i];r[i]=(uint64_t)c;c>>=64;}}
    }
}

/* ============================================================
 * ALL-GPU kernel: midstate SHA-256d + scalar mul + EC + SHA-256 + double hash DER
 *
 * Inputs:
 *   d_midstate[8]:    SHA-256 state after processing first 78 blocks (4992 bytes)
 *   d_tail[16]:       bytes 4992..5007 of preimage (last 8 of prefix + locktime placeholder + sighash_type)
 *   tail_offset:      offset within tail where locktime goes (= prefix_len - 4992)
 *   total_len:        total preimage length (prefix_len + 8)
 *   d_neg_r_inv[4]:   precomputed -r^{-1} mod N
 *   d_u2rx[4], d_u2ry[4]: precomputed u2*R point
 * ============================================================ */

__global__ void kernel_allgpu_pinning(
    const uint32_t *d_midstate,
    const uint8_t *d_suffix,      /* suffix template with placeholders */
    int suffix_len,
    int seq_offset,               /* offset of sequence in suffix */
    int lt_offset,                /* offset of locktime in suffix */
    int total_preimage_len,
    uint32_t seq_value,
    uint32_t start_lt,
    const uint64_t *d_neg_r_inv,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gtX, uint8_t *d_gtY,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
    int batch_size, int easy_mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    uint32_t lt = start_lt + (uint32_t)idx;

    /* Copy suffix, set sequence + locktime */
    uint8_t buf[128];
    for(int i=0;i<suffix_len;i++) buf[i]=d_suffix[i];
    buf[seq_offset]=(seq_value)&0xFF; buf[seq_offset+1]=(seq_value>>8)&0xFF;
    buf[seq_offset+2]=(seq_value>>16)&0xFF; buf[seq_offset+3]=(seq_value>>24)&0xFF;
    buf[lt_offset]=(lt)&0xFF; buf[lt_offset+1]=(lt>>8)&0xFF;
    buf[lt_offset+2]=(lt>>16)&0xFF; buf[lt_offset+3]=(lt>>24)&0xFF;

    /* SHA-256 padding */
    buf[suffix_len]=0x80;
    for(int i=suffix_len+1;i<128;i++) buf[i]=0;
    int nblk=(suffix_len<56)?1:2;
    uint64_t bit_len=(uint64_t)total_preimage_len*8;
    int last=nblk*64-8;
    buf[last]=(bit_len>>56)&0xFF;buf[last+1]=(bit_len>>48)&0xFF;
    buf[last+2]=(bit_len>>40)&0xFF;buf[last+3]=(bit_len>>32)&0xFF;
    buf[last+4]=(bit_len>>24)&0xFF;buf[last+5]=(bit_len>>16)&0xFF;
    buf[last+6]=(bit_len>>8)&0xFF;buf[last+7]=bit_len&0xFF;

    uint32_t state[8]; for(int i=0;i<8;i++) state[i]=d_midstate[i];
    for(int b=0;b<nblk;b++){
        uint32_t blk[16]; for(int i=0;i<16;i++)
            blk[i]=((uint32_t)buf[b*64+i*4]<<24)|((uint32_t)buf[b*64+i*4+1]<<16)|
                   ((uint32_t)buf[b*64+i*4+2]<<8)|(uint32_t)buf[b*64+i*4+3];
        gpu_sha256_compress_block(state,blk);
    }

    /* Second SHA-256 */
    uint8_t first[32]; for(int i=0;i<8;i++){first[i*4]=(state[i]>>24)&0xFF;first[i*4+1]=(state[i]>>16)&0xFF;
        first[i*4+2]=(state[i]>>8)&0xFF;first[i*4+3]=state[i]&0xFF;}
    uint8_t p2[64]; memset(p2,0,64); memcpy(p2,first,32); p2[32]=0x80; p2[62]=0x01; p2[63]=0x00;
    uint32_t b2[16]; for(int i=0;i<16;i++) b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
        ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    gpu_sha256_compress_block(s2,b2);

    uint8_t sighash[32]; for(int i=0;i<8;i++){sighash[i*4]=(s2[i]>>24)&0xFF;sighash[i*4+1]=(s2[i]>>16)&0xFF;
        sighash[i*4+2]=(s2[i]>>8)&0xFF;sighash[i*4+3]=s2[i]&0xFF;}
    uint64_t z[4]; for(int i=0;i<4;i++){z[i]=0;for(int b=0;b<8;b++)z[i]|=(uint64_t)sighash[31-i*8-b]<<(b*8);}
    uint64_t nri[4]={d_neg_r_inv[0],d_neg_r_inv[1],d_neg_r_inv[2],d_neg_r_inv[3]};
    uint64_t u1[4]; gpu_scalar_mulmod(u1,nri,z);
    uint16_t pk[16]; memcpy(pk,u1,32);
    uint64_t qx[4],qy[4]; _PointMultiSecp256k1(qx,qy,pk,d_gtX,d_gtY);

    /* Q1 = u1*G + u2*R (recovery flag 0, even y) */
    uint64_t u2rx[4]={d_u2rx[0],d_u2rx[1],d_u2rx[2],d_u2rx[3]};
    uint64_t u2ry[4]={d_u2ry[0],d_u2ry[1],d_u2ry[2],d_u2ry[3]};
    uint64_t q1x[4],q1y[4],q1z[5];
    memcpy(q1x,qx,32); memcpy(q1y,qy,32);
    q1z[0]=1;q1z[1]=0;q1z[2]=0;q1z[3]=0;q1z[4]=0;
    _PointAddSecp256k1(q1x,q1y,q1z,u2rx,u2ry);

    /* Q2 = Q1 + neg_2u2R = u1*G - u2*R (recovery flag 1, odd y) */
    uint64_t q2x[4],q2y[4],q2z[5];
    memcpy(q2x,q1x,32); memcpy(q2y,q1y,32); memcpy(q2z,q1z,40);
    uint64_t n2rx[4]={d_neg2u2rx[0],d_neg2u2rx[1],d_neg2u2rx[2],d_neg2u2rx[3]};
    uint64_t n2ry[4]={d_neg2u2ry[0],d_neg2u2ry[1],d_neg2u2ry[2],d_neg2u2ry[3]};
    _PointAddSecp256k1(q2x,q2y,q2z,n2rx,n2ry);

    /* Batch ModInv: invert q1z and q2z with one inversion */
    uint64_t prod[5];
    _ModMult(prod,q1z,q2z);      /* prod = q1z * q2z */
    _ModInv(prod);                /* prod = 1/(q1z*q2z) */
    uint64_t inv1[5],inv2[5];
    _ModMult(inv1,prod,q2z);      /* inv1 = q2z/(q1z*q2z) = 1/q1z */
    _ModMult(inv2,prod,q1z);      /* inv2 = q1z/(q1z*q2z) = 1/q2z */

    /* Normalize Q1 */
    uint64_t zz[4],zzz[4];
    _ModMult(zz,inv1,inv1);_ModMult(zzz,zz,inv1);
    _ModMult(q1x,zz);_ModMult(q1y,zzz);

    /* Normalize Q2 */
    _ModMult(zz,inv2,inv2);_ModMult(zzz,zz,inv2);
    _ModMult(q2x,zz);_ModMult(q2y,zzz);

    /* Check Q1 (recid=0): SHA-256 + double hash */
    int v=0, hash_choice=0, recid=0;
    {
        uint32_t *x32=(uint32_t*)q1x;
        uint32_t pb[16];
        pb[0]=__byte_perm(x32[7],0x2+(uint8_t)(q1y[0]&1),0x4321);
        pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
        pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
        pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
        pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8]; _SHA256Initialize(hs); _SHA256Transform(hs,pb);
        uint8_t h[32]; for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
            h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
        v=easy_mode?gpu_is_der_easy(h,32):gpu_is_valid_der(h,32);
        if(!v){
            uint8_t p2[64];memset(p2,0,64);memcpy(p2,h,32);p2[32]=0x80;p2[62]=1;p2[63]=0;
            uint32_t b2[16];for(int i=0;i<16;i++)b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
                ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
            uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,b2);
            uint8_t h2[32];for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
                h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
            v=easy_mode?gpu_is_der_easy(h2,32):gpu_is_valid_der(h2,32);
            hash_choice=1;
        }
    }

    /* Check Q2 (recid=1) if Q1 missed */
    if(!v){
        hash_choice=0; recid=1;
        uint32_t *x32=(uint32_t*)q2x;
        uint32_t pb[16];
        pb[0]=__byte_perm(x32[7],0x2+(uint8_t)(q2y[0]&1),0x4321);
        pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
        pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
        pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
        pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8]; _SHA256Initialize(hs); _SHA256Transform(hs,pb);
        uint8_t h[32]; for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
            h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
        v=easy_mode?gpu_is_der_easy(h,32):gpu_is_valid_der(h,32);
        if(!v){
            uint8_t p2[64];memset(p2,0,64);memcpy(p2,h,32);p2[32]=0x80;p2[62]=1;p2[63]=0;
            uint32_t b2[16];for(int i=0;i<16;i++)b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
                ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
            uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,b2);
            uint8_t h2[32];for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
                h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
            v=easy_mode?gpu_is_der_easy(h2,32):gpu_is_valid_der(h2,32);
            hash_choice=1;
        }
    }

    /* Hit: encode idx(29:0) | recid(30) | hash_choice(31) */
    if(v){uint32_t pos=atomicAdd(d_hit_cnt,1);
        if(pos<1024)d_hit_idx[pos]=((uint32_t)idx)|(recid<<30)|(hash_choice<<31);}
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

static void compute_gtable(uint8_t *gTableX, uint8_t *gTableY) {
    size_t gt_bytes = 16ULL * 65536 * 32;
    const char *cache = "/tmp/secp256k1_gtable.bin";
    
    /* Try loading from cache */
    FILE *f = fopen(cache, "rb");
    if (f) {
        size_t r1 = fread(gTableX, 1, gt_bytes, f);
        size_t r2 = fread(gTableY, 1, gt_bytes, f);
        fclose(f);
        if (r1 == gt_bytes && r2 == gt_bytes) {
            printf("  GTable loaded from cache (%s)\n", cache);
            return;
        }
        printf("  GTable cache corrupt, recomputing...\n");
    }
    
    printf("  Computing GTable (first run, ~5 min)...\n");
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*shift=BN_new();
    EC_POINT *base=EC_POINT_new(grp),*pt=EC_POINT_new(grp);
    EC_POINT_copy(base, EC_GROUP_get0_generator(grp));
    for(int chunk=0;chunk<16;chunk++){
        if(chunk>0){BN_set_word(shift,65536);EC_POINT_mul(grp,base,NULL,base,shift,ctx);}
        EC_POINT_copy(pt,base);
        for(int j=0;j<65535;j++){
            int elem=chunk*65536+j;
            EC_POINT_get_affine_coordinates_GFp(grp,pt,x,y,ctx);
            uint8_t xbe[32],ybe[32]; memset(xbe,0,32);memset(ybe,0,32);
            BN_bn2bin(x,xbe+(32-BN_num_bytes(x)));
            BN_bn2bin(y,ybe+(32-BN_num_bytes(y)));
            for(int b=0;b<32;b++){gTableX[elem*32+b]=xbe[31-b];gTableY[elem*32+b]=ybe[31-b];}
            EC_POINT_add(grp,pt,pt,base,ctx);
        }
        printf("    Chunk %d/16\n",chunk+1);
    }
    BN_free(x);BN_free(y);BN_free(shift);
    EC_POINT_free(base);EC_POINT_free(pt);
    BN_CTX_free(ctx);EC_GROUP_free(grp);
    
    /* Save to cache */
    f = fopen(cache, "wb");
    if (f) {
        fwrite(gTableX, 1, gt_bytes, f);
        fwrite(gTableY, 1, gt_bytes, f);
        fclose(f);
        printf("  GTable saved to cache (%s)\n", cache);
    }
    printf("  GTable done.\n");
}

/* CPU: compute SHA-256 midstate for first N full blocks */
static void compute_midstate(uint32_t midstate[8], const uint8_t *data, int full_blocks) {
    /* Initialize SHA-256 state */
    midstate[0]=0x6a09e667; midstate[1]=0xbb67ae85;
    midstate[2]=0x3c6ef372; midstate[3]=0xa54ff53a;
    midstate[4]=0x510e527f; midstate[5]=0x9b05688c;
    midstate[6]=0x1f83d9ab; midstate[7]=0x5be0cd19;
    
    /* Process each 64-byte block */
    for (int b = 0; b < full_blocks; b++) {
        const uint8_t *blk = data + b * 64;
        /* Convert to big-endian uint32 and compress */
        SHA256_CTX tmp;
        SHA256_Init(&tmp);
        /* Hack: use OpenSSL to process blocks. Actually let's do it manually. */
        
        uint32_t W[16];
        for (int i = 0; i < 16; i++)
            W[i] = ((uint32_t)blk[i*4]<<24)|((uint32_t)blk[i*4+1]<<16)|
                   ((uint32_t)blk[i*4+2]<<8)|(uint32_t)blk[i*4+3];
        
        /* Inline SHA-256 compress */
        uint32_t w[64];
        for(int i=0;i<16;i++) w[i]=W[i];
        for(int i=16;i<64;i++){
            uint32_t s0=((w[i-15]>>7)|(w[i-15]<<25))^((w[i-15]>>18)|(w[i-15]<<14))^(w[i-15]>>3);
            uint32_t s1=((w[i-2]>>17)|(w[i-2]<<15))^((w[i-2]>>19)|(w[i-2]<<13))^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        static const uint32_t K[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t a=midstate[0],bb=midstate[1],c=midstate[2],d=midstate[3];
        uint32_t e=midstate[4],f=midstate[5],g=midstate[6],h=midstate[7];
        for(int i=0;i<64;i++){
            uint32_t S1=((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7));
            uint32_t ch=(e&f)^(~e&g);
            uint32_t t1=h+S1+ch+K[i]+w[i];
            uint32_t S0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));
            uint32_t maj=(a&bb)^(a&c)^(bb&c);
            uint32_t t2=S0+maj;
            h=g;g=f;f=e;e=d+t1;d=c;c=bb;bb=a;a=t1+t2;
        }
        midstate[0]+=a;midstate[1]+=bb;midstate[2]+=c;midstate[3]+=d;
        midstate[4]+=e;midstate[5]+=f;midstate[6]+=g;midstate[7]+=h;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("Usage: %s bench\n", argv[0]); return 1; }
    
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("QSB All-GPU Pinning (midstate trick)\n");
    printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    
    /* GTable */
    size_t gt_sz = 16ULL*65536*32;
    uint8_t *h_gtX=(uint8_t*)malloc(gt_sz), *h_gtY=(uint8_t*)malloc(gt_sz);
    compute_gtable(h_gtX, h_gtY);
    uint8_t *d_gtX, *d_gtY;
    cudaMalloc(&d_gtX,gt_sz); cudaMalloc(&d_gtY,gt_sz);
    cudaMemcpy(d_gtX,h_gtX,gt_sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_gtY,h_gtY,gt_sz,cudaMemcpyHostToDevice);
    free(h_gtX); free(h_gtY);
    
    /* Signature */
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bn_ctx = BN_CTX_new(); BIGNUM *bn_n = BN_new();
    EC_GROUP_get_order(grp, bn_n, bn_ctx);
    EC_KEY *ec_key = EC_KEY_new(); EC_KEY_set_group(ec_key,grp); EC_KEY_generate_key(ec_key);
    uint8_t msg[32]; SHA256((uint8_t*)"qsb_allgpu",10,msg);
    ECDSA_SIG *sig = ECDSA_do_sign(msg,32,ec_key);
    const BIGNUM *sr,*ss; ECDSA_SIG_get0(sig,&sr,&ss);
    uint8_t sig_r[32],sig_s[32]; memset(sig_r,0,32);memset(sig_s,0,32);
    BN_bn2bin(sr,sig_r+(32-BN_num_bytes(sr))); BN_bn2bin(ss,sig_s+(32-BN_num_bytes(ss)));
    
    BIGNUM *bn_r=BN_new(),*bn_s=BN_new(),*bn_ri=BN_new(),*bn_u2=BN_new();
    BN_bin2bn(sig_r,32,bn_r); BN_bin2bn(sig_s,32,bn_s);
    BN_mod_inverse(bn_ri,bn_r,bn_n,bn_ctx);
    BN_mod_mul(bn_u2,bn_s,bn_ri,bn_n,bn_ctx);
    
    /* neg_r_inv */
    uint8_t ri_be[32]; memset(ri_be,0,32);
    BN_bn2bin(bn_ri,ri_be+(32-BN_num_bytes(bn_ri)));
    uint64_t ri_le[4]; be_to_le64(ri_le, ri_be);
    uint64_t nri_le[4];
    { /* N - ri */
        __int128 c=0;
        const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
        for(int i=0;i<4;i++){c+=(__int128)N[i]-ri_le[i];nri_le[i]=(uint64_t)c;c>>=64;}
    }
    
    /* u2R */
    EC_POINT *R_pt=EC_POINT_new(grp);
    EC_POINT_set_compressed_coordinates_GFp(grp,R_pt,bn_r,0,bn_ctx);
    EC_POINT *u2R=EC_POINT_new(grp);
    EC_POINT_mul(grp,u2R,NULL,R_pt,bn_u2,bn_ctx);
    BIGNUM *ux=BN_new(),*uy=BN_new();
    EC_POINT_get_affine_coordinates_GFp(grp,u2R,ux,uy,bn_ctx);
    uint8_t ux_be[32],uy_be[32]; memset(ux_be,0,32);memset(uy_be,0,32);
    BN_bn2bin(ux,ux_be+(32-BN_num_bytes(ux))); BN_bn2bin(uy,uy_be+(32-BN_num_bytes(uy)));
    uint64_t h_u2rx[4],h_u2ry[4]; be_to_le64(h_u2rx,ux_be); be_to_le64(h_u2ry,uy_be);
    
    /* neg_2u2R = -(2 * u2R) for second recovery flag */
    EC_POINT *neg2u2R=EC_POINT_new(grp);
    EC_POINT_dbl(grp, neg2u2R, u2R, bn_ctx);    /* 2*u2R */
    EC_POINT_invert(grp, neg2u2R, bn_ctx);       /* negate */
    BIGNUM *n2x=BN_new(),*n2y=BN_new();
    EC_POINT_get_affine_coordinates_GFp(grp,neg2u2R,n2x,n2y,bn_ctx);
    uint8_t n2x_be[32],n2y_be[32]; memset(n2x_be,0,32);memset(n2y_be,0,32);
    BN_bn2bin(n2x,n2x_be+(32-BN_num_bytes(n2x))); BN_bn2bin(n2y,n2y_be+(32-BN_num_bytes(n2y)));
    uint64_t h_neg2u2rx[4],h_neg2u2ry[4]; be_to_le64(h_neg2u2rx,n2x_be); be_to_le64(h_neg2u2ry,n2y_be);
    
    /* Upload constants */
    uint64_t *d_nri, *d_u2rx, *d_u2ry, *d_neg2u2rx, *d_neg2u2ry;
    cudaMalloc(&d_nri,32); cudaMalloc(&d_u2rx,32); cudaMalloc(&d_u2ry,32);
    cudaMalloc(&d_neg2u2rx,32); cudaMalloc(&d_neg2u2ry,32);
    cudaMemcpy(d_nri,nri_le,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2rx,h_u2rx,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2ry,h_u2ry,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neg2u2rx,h_neg2u2rx,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_neg2u2ry,h_neg2u2ry,32,cudaMemcpyHostToDevice);
    
    /* Build fake sighash preimage:
     * fixed_prefix (5000 bytes, covers scriptCode) → midstate
     * suffix: sc_tail(8) + sequence(4) + outputs(38) + locktime(4) + sighash_type(4) = 58 bytes
     */
    int scriptcode_prefix_len = 5000;  /* midstate covers this */
    uint8_t *fixed_prefix = (uint8_t*)calloc(scriptcode_prefix_len, 1);
    for(int i=0;i<scriptcode_prefix_len;i++) fixed_prefix[i]=(uint8_t)(i*37+13);
    
    int full_blocks = scriptcode_prefix_len / 64;
    int sc_tail_len = scriptcode_prefix_len - full_blocks * 64;  /* remainder after midstate */
    
    /* Build suffix template */
    int output_len = 38;  /* 1(count) + 8(value) + 1(script_len) + 25(p2pkh) + 3(padding) */
    int suffix_len = sc_tail_len + 4 + output_len + 4 + 4;  /* sc_tail + seq + outputs + lt + sighash */
    int seq_offset = sc_tail_len;
    int lt_offset = sc_tail_len + 4 + output_len;
    int total_preimage_len = scriptcode_prefix_len + 4 + output_len + 4 + 4;
    
    uint8_t *suffix_template = (uint8_t*)calloc(128, 1);
    /* Copy scriptCode tail */
    memcpy(suffix_template, fixed_prefix + full_blocks * 64, sc_tail_len);
    /* Sequence placeholder at seq_offset (will be set by GPU) */
    /* Fake output data */
    for(int i=0;i<output_len;i++) suffix_template[seq_offset+4+i]=(uint8_t)(i*53+7);
    /* Locktime placeholder at lt_offset (will be set by GPU) */
    /* Sighash type */
    suffix_template[lt_offset+4] = 0x01;
    suffix_template[lt_offset+5] = 0x00;
    suffix_template[lt_offset+6] = 0x00;
    suffix_template[lt_offset+7] = 0x00;
    
    printf("  Midstate: %d blocks (%d bytes), sc_tail=%d bytes\n", full_blocks, full_blocks*64, sc_tail_len);
    printf("  Suffix: %d bytes (seq@%d, lt@%d)\n", suffix_len, seq_offset, lt_offset);
    printf("  Total preimage: %d bytes\n", total_preimage_len);
    printf("  GPU: %d SHA-256 blocks per candidate + 1 for double-hash\n", (suffix_len<56)?1:2);
    
    uint32_t midstate[8];
    compute_midstate(midstate, fixed_prefix, full_blocks);
    
    /* Upload to GPU */
    uint32_t *d_midstate; cudaMalloc(&d_midstate, 32);
    uint8_t *d_suffix; cudaMalloc(&d_suffix, 128);
    cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix, suffix_template, 128, cudaMemcpyHostToDevice);
    
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    uint32_t *d_hit_cnt, *d_hit_idx;
    cudaMalloc(&d_hit_cnt,4); cudaMalloc(&d_hit_idx,1024*4);
    
    int BATCH = 262144;
    int BLKSZ = 256;
    int GRDSZ = (BATCH+BLKSZ-1)/BLKSZ;
    
    if (strcmp(argv[1], "bench") == 0) {
        printf("\n  --- All-GPU benchmark (%d per batch) ---\n", BATCH);
        
        /* Warmup */
        uint32_t h_hit=0; cudaMemcpy(d_hit_cnt,&h_hit,4,cudaMemcpyHostToDevice);
        kernel_allgpu_pinning<<<GRDSZ,BLKSZ>>>(
            d_midstate, d_suffix, suffix_len, seq_offset, lt_offset, total_preimage_len, 0, 0,
            d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry, d_gtX, d_gtY,
            d_hit_cnt, d_hit_idx, BATCH, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
        cudaMemcpy(&h_hit,d_hit_cnt,4,cudaMemcpyDeviceToHost);
        printf("  Warmup: %u hits (~1/8 double hash)\n", h_hit);
        
        /* Timed */
        h_hit=0; cudaMemcpy(d_hit_cnt,&h_hit,4,cudaMemcpyHostToDevice);
        cudaEvent_t ev0,ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        int total=0, nb=40;
        cudaEventRecord(ev0);
        for(int b=0;b<nb;b++){
            kernel_allgpu_pinning<<<GRDSZ,BLKSZ>>>(
                d_midstate, d_suffix, suffix_len, seq_offset, lt_offset, total_preimage_len, 0, (uint32_t)(b*BATCH),
                d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry, d_gtX, d_gtY,
                d_hit_cnt, d_hit_idx, BATCH, 1);
            total += BATCH;
        }
        cudaEventRecord(ev1); cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
        
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
        cudaMemcpy(&h_hit,d_hit_cnt,4,cudaMemcpyDeviceToHost);
        double rate = total/(ms/1000.0);
        
        printf("  %d candidates in %.1f ms\n", total, ms);
        printf("  Rate: %.0f/s (%.1f μs each)\n", rate, ms*1000.0/total);
        printf("  Hits (~1/8 double hash): %u\n", h_hit);
        
        double target=pow(2,43.4);  /* 4 DER checks/cand: 2 recovery flags × 2 hashes = 2^-43.4 */
        double hours=target/rate/3600.0;
        double cost=0.089;
        printf("\n  === EXTRAPOLATION (pinning, 4 checks/cand, 2^43.4) ===\n");
        printf("  Rate: %.0f/s\n", rate);
        printf("  This machine: %.0f hours (%.1f days), $%.0f\n", hours, hours/24, hours*cost);
        for(int n=10;n<=100;n*=10)
            printf("  %dx: %.0fh wall, $%.0f total\n", n, hours/n, hours*cost);
        
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    
    } else if (strcmp(argv[1],"search")==0) {
        /* Multi-GPU search: ./qsb_allgpu search <start_seq> <num_seqs> [easy]
         * Each sequence: sweep all 2^32 locktimes */
        uint32_t start_seq = (argc >= 3) ? (uint32_t)atol(argv[2]) : 0;
        uint32_t num_seqs = (argc >= 4) ? (uint32_t)atol(argv[3]) : 100000;
        int easy = (argc >= 5 && strcmp(argv[4],"easy")==0) ? 1 : 0;
        
        printf("\n  === Pinning Search (seq %u..%u, all locktimes, %s) ===\n",
               start_seq, start_seq + num_seqs - 1, easy?"EASY":"REAL");
        printf("  %u sequences × 2^32 locktimes = %.2e candidates\n",
               num_seqs, (double)num_seqs * (1ULL<<32));
        
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t total_searched = 0;
        int found = 0;
        
        for (uint32_t seq = start_seq; seq < start_seq + num_seqs && !found; seq++) {
            /* Sweep all 2^32 locktimes for this sequence */
            uint64_t lt_searched = 0;
            while (lt_searched < (1ULL << 32) && !found) {
                uint32_t batch_lt = (uint32_t)lt_searched;
                uint32_t h_hit = 0;
                cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);
                
                kernel_allgpu_pinning<<<GRDSZ,BLKSZ>>>(
                    d_midstate, d_suffix, suffix_len, seq_offset, lt_offset,
                    total_preimage_len, seq, batch_lt,
                    d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry, d_gtX, d_gtY,
                    d_hit_cnt, d_hit_idx, BATCH, easy);
                cudaDeviceSynchronize();
                
                lt_searched += BATCH;
                total_searched += BATCH;
                
                cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
                if (h_hit > 0) {
                    uint32_t hit_indices[16];
                    int nh = (h_hit > 16) ? 16 : h_hit;
                    cudaMemcpy(hit_indices, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);
                    
                    printf("\n  *** HIT! seq=%u, %u hits ***\n", seq, h_hit);
                    mkdir("results", 0755);
                    char fname[256];
                    snprintf(fname, sizeof(fname), "results/pinning_hit.txt");
                    FILE *f = fopen(fname, "w");
                    if (f) {
                        for (int h = 0; h < nh; h++) {
                            uint32_t raw = hit_indices[h];
                            uint32_t lt = batch_lt + (raw & 0x3FFFFFFF);
                            int ri = (raw >> 30) & 1;
                            int hc = (raw >> 31) & 1;
                            fprintf(f, "sequence=%u\nlocktime=%u\nhash_choice=%d\nrecid=%d\n", seq, lt, hc, ri);
                            printf("  sequence=%u locktime=%u hash_choice=%d recid=%d (%s, %s)\n",
                                   seq, lt, hc, ri,
                                   hc ? "SHA256(SHA256)" : "SHA256",
                                   ri ? "odd_y" : "even_y");
                        }
                        fclose(f);
                    }
                    found = 1;
                }
            }
            
            /* Progress per sequence */
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            if ((seq - start_seq) % 10 == 0 || found) {
                printf("  seq=%u (%u/%u), total=%luM, %.1fM/s, %.0fs\n",
                       seq, seq-start_seq+1, num_seqs,
                       total_searched/1000000, total_searched/elapsed/1e6, elapsed);
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        printf("\n  Done: %luM in %.0fs (%.1fM/s), found=%d\n",
               total_searched/1000000, elapsed, total_searched/elapsed/1e6, found);
    }
    
    /* Cleanup */
    cudaFree(d_gtX);cudaFree(d_gtY);cudaFree(d_nri);cudaFree(d_u2rx);cudaFree(d_u2ry);
    cudaFree(d_neg2u2rx);cudaFree(d_neg2u2ry);
    cudaFree(d_midstate);cudaFree(d_suffix);cudaFree(d_hit_cnt);cudaFree(d_hit_idx);
    free(fixed_prefix);free(suffix_template);
    BN_free(bn_r);BN_free(bn_s);BN_free(bn_ri);BN_free(bn_u2);BN_free(ux);BN_free(uy);BN_free(n2x);BN_free(n2y);BN_free(bn_n);
    EC_POINT_free(R_pt);EC_POINT_free(u2R);EC_POINT_free(neg2u2R);EC_GROUP_free(grp);BN_CTX_free(bn_ctx);
    EC_KEY_free(ec_key);ECDSA_SIG_free(sig);
    return 0;
}
