/* qsb_real_search.cu — Real pinning search with sequence + locktime variation
 *
 * Reads pinning2.bin (midstate with sequence in suffix)
 * Loops: outer=sequence (0x80000000+), inner=locktime (500000000-1744600000)
 * 4 DER checks per candidate (2 recovery flags × 2 hashes)
 *
 * Build:  nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm
 * Usage:  ./qsb_real pinning2.bin [easy]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

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

/* GTable lookup */
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

/* Check if x is a valid x-coordinate on secp256k1 (y² = x³+7 has a root mod p) */
__device__ int gpu_is_on_curve(uint64_t *x) {
    uint64_t x2[4], x3[4];
    _ModSqr(x2, x);
    _ModMult(x3, x2, x);
    /* y_sq = x³ + 7 mod p. Addition with carry; result reduced if >= p. */
    uint64_t y_sq[4];
    uint64_t c;
    y_sq[0] = x3[0] + 7ULL; c = (y_sq[0] < x3[0]);
    y_sq[1] = x3[1] + c; c = (y_sq[1] < x3[1]);
    y_sq[2] = x3[2] + c; c = (y_sq[2] < x3[2]);
    y_sq[3] = x3[3] + c;
    /* If y_sq >= p, subtract p. p = 2^256 - 2^32 - 977, so y_sq + 977 + 2^32 wraps */
    /* Since x³ < p, y_sq < p+7, at most one subtraction needed */
    const uint64_t P_LO = 0xFFFFFFFEFFFFFC2FULL;
    if (y_sq[3] == 0xFFFFFFFFFFFFFFFFULL && y_sq[2] == 0xFFFFFFFFFFFFFFFFULL
        && y_sq[1] == 0xFFFFFFFFFFFFFFFFULL && y_sq[0] >= P_LO) {
        /* y_sq -= p */
        uint64_t t = y_sq[0] - P_LO; y_sq[0] = t;
        /* upper words: subtracting 0xFFFFFFFF... means they become 0 */
        y_sq[1] = 0; y_sq[2] = 0; y_sq[3] = 0;
    }

    /* y = y_sq^((p+1)/4) mod p using square-and-multiply.
     * (p+1)/4 = 0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C
     * 254-bit exponent. */
    const uint64_t EXP[4] = {
        0xFFFFFFFFBFFFFF0CULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x3FFFFFFFFFFFFFFFULL
    };
    uint64_t y[4] = {1, 0, 0, 0};
    for (int i = 253; i >= 0; i--) {
        uint64_t tmp[4];
        _ModSqr(tmp, y);
        y[0]=tmp[0]; y[1]=tmp[1]; y[2]=tmp[2]; y[3]=tmp[3];
        int bit = (EXP[i / 64] >> (i % 64)) & 1;
        if (bit) {
            _ModMult(tmp, y, y_sq);
            y[0]=tmp[0]; y[1]=tmp[1]; y[2]=tmp[2]; y[3]=tmp[3];
        }
    }
    uint64_t y2[4];
    _ModSqr(y2, y);
    return y2[0]==y_sq[0] && y2[1]==y_sq[1] && y2[2]==y_sq[2] && y2[3]==y_sq[3];
}

/* Given a DER signature, check if its r value is a valid x-coord on secp256k1.
 * Tries both r and r+n (since verification check is mod n). */
__device__ int gpu_der_r_on_curve(const uint8_t *der) {
    int rl = der[3];
    int r_start = 4;
    if (rl > 0 && der[r_start] == 0) { r_start++; rl--; }
    if (rl > 32 || rl <= 0) return 0;

    uint8_t rbe[32] = {0};
    for (int i = 0; i < rl; i++) rbe[32 - rl + i] = der[r_start + i];

    uint64_t r[4];
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) v |= (uint64_t)rbe[31 - i*8 - b] << (b*8);
        r[i] = v;
    }

    if (gpu_is_on_curve(r)) return 1;

    /* Try r + n < p */
    const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,
                         0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t rn[4]; uint64_t c;
    uint64_t t = r[0] + N[0]; c = (t < r[0]); rn[0] = t;
    t = r[1] + N[1] + c; c = (t < r[1]) || (c && t == r[1]); rn[1] = t;
    t = r[2] + N[2] + c; c = (t < r[2]) || (c && t == r[2]); rn[2] = t;
    t = r[3] + N[3] + c; if (t < r[3] || (c && t == r[3])) return 0; /* overflow */
    rn[3] = t;
    /* rn < p? */
    const uint64_t P_LO = 0xFFFFFFFEFFFFFC2FULL;
    int rn_lt_p = (rn[3] != 0xFFFFFFFFFFFFFFFFULL) ||
                  (rn[2] != 0xFFFFFFFFFFFFFFFFULL) ||
                  (rn[1] != 0xFFFFFFFFFFFFFFFFULL) ||
                  (rn[0] < P_LO);
    if (!rn_lt_p) return 0;
    return gpu_is_on_curve(rn);
}

/* 256-bit modular multiplication for scalar */
/* Correct modular multiplication mod secp256k1 group order n
 * Uses reduction: 2^256 ≡ c (mod n) where c = 2^256 - n (129 bits) */
__device__ void gpu_scalar_mulmod(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    /* Step 1: 512-bit schoolbook multiplication */
    __int128 t[8]={0};
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) t[i+j]+=(__int128)a[i]*b[j];
    for(int i=0;i<7;i++){t[i+1]+=t[i]>>64;t[i]&=0xFFFFFFFFFFFFFFFFULL;}
    uint64_t p[8]; for(int i=0;i<8;i++) p[i]=(uint64_t)t[i];

    /* c = 2^256 - n = {C0, C1, 1, 0} */
    const uint64_t C0=0x402DA1732FC9BEBFULL, C1=0x4551231950B75FC4ULL;

    /* Step 2: first reduction — p_hi * c + p_lo */
    __int128 acc[8]={0};
    for(int i=0;i<4;i++) acc[i]+=(__int128)p[4+i]*C0;
    for(int i=0;i<4;i++) acc[i+1]+=(__int128)p[4+i]*C1;
    for(int i=0;i<4;i++) acc[i+2]+=(__int128)p[4+i];
    for(int i=0;i<4;i++) acc[i]+=p[i];
    for(int i=0;i<7;i++){acc[i+1]+=acc[i]>>64;acc[i]&=0xFFFFFFFFFFFFFFFFULL;}
    uint64_t q[8]; for(int i=0;i<8;i++) q[i]=(uint64_t)acc[i];

    /* Step 3: second reduction — q_hi * c + q_lo */
    __int128 acc2[8]={0};
    for(int i=0;i<4;i++) acc2[i]+=(__int128)q[4+i]*C0;
    for(int i=0;i<4;i++) acc2[i+1]+=(__int128)q[4+i]*C1;
    for(int i=0;i<4;i++) acc2[i+2]+=(__int128)q[4+i];
    for(int i=0;i<4;i++) acc2[i]+=q[i];
    for(int i=0;i<7;i++){acc2[i+1]+=acc2[i]>>64;acc2[i]&=0xFFFFFFFFFFFFFFFFULL;}
    uint64_t res[5]; for(int i=0;i<5;i++) res[i]=(uint64_t)acc2[i];

    /* Step 4: conditional subtraction of n */
    const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    for(int rep=0;rep<3;rep++){
        int ge=0;
        if(res[4]>0) ge=1;
        else{ ge=1; for(int i=3;i>=0;i--){if(res[i]>N[i]){ge=1;break;}if(res[i]<N[i]){ge=0;break;}} }
        if(!ge) break;
        __int128 borrow=0;
        for(int i=0;i<4;i++){borrow+=(__int128)res[i]-N[i];res[i]=(uint64_t)borrow;borrow>>=64;}
        res[4]+=(uint64_t)borrow;
    }
    for(int i=0;i<4;i++) r[i]=res[i];
}

/* ============================================================
 * Kernel: searches locktime range for a fixed sequence value
 * ============================================================ */

__global__ void kernel_pinning_real(
    const uint32_t *d_midstate,
    const uint8_t *d_suffix,    /* suffix template */
    int suffix_len,             /* total suffix including lt+sighash */
    int seq_offset,             /* offset of sequence in suffix */
    int lt_offset,              /* offset of locktime in suffix */
    int total_preimage_len,
    uint32_t seq_value,         /* current sequence value */
    uint32_t start_lt,          /* starting locktime for this batch */
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
    uint8_t buf[192];
    for(int i=0;i<suffix_len;i++) buf[i]=d_suffix[i];
    buf[seq_offset]=(seq_value)&0xFF; buf[seq_offset+1]=(seq_value>>8)&0xFF;
    buf[seq_offset+2]=(seq_value>>16)&0xFF; buf[seq_offset+3]=(seq_value>>24)&0xFF;
    buf[lt_offset]=(lt)&0xFF; buf[lt_offset+1]=(lt>>8)&0xFF;
    buf[lt_offset+2]=(lt>>16)&0xFF; buf[lt_offset+3]=(lt>>24)&0xFF;

    /* SHA-256 padding */
    buf[suffix_len]=0x80;
    for(int i=suffix_len+1;i<192;i++) buf[i]=0;
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
        _SHA256Transform(state,blk);
    }

    /* Second SHA-256 */
    uint8_t first[32]; for(int i=0;i<8;i++){first[i*4]=(state[i]>>24)&0xFF;first[i*4+1]=(state[i]>>16)&0xFF;
        first[i*4+2]=(state[i]>>8)&0xFF;first[i*4+3]=state[i]&0xFF;}
    uint8_t p2[64]; memset(p2,0,64); memcpy(p2,first,32); p2[32]=0x80; p2[62]=0x01; p2[63]=0x00;
    uint32_t b2[16]; for(int i=0;i<16;i++) b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
        ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2,b2);

    uint8_t sighash[32]; for(int i=0;i<8;i++){sighash[i*4]=(s2[i]>>24)&0xFF;sighash[i*4+1]=(s2[i]>>16)&0xFF;
        sighash[i*4+2]=(s2[i]>>8)&0xFF;sighash[i*4+3]=s2[i]&0xFF;}

    /* Scalar mul */
    uint64_t z[4]; for(int i=0;i<4;i++){z[i]=0;for(int b=0;b<8;b++)z[i]|=(uint64_t)sighash[31-i*8-b]<<(b*8);}
    uint64_t nri[4]={d_neg_r_inv[0],d_neg_r_inv[1],d_neg_r_inv[2],d_neg_r_inv[3]};
    uint64_t u1[4]; gpu_scalar_mulmod(u1,nri,z);
    uint16_t pk[16]; memcpy(pk,u1,32);
    uint64_t qx[4],qy[4]; _PointMultiSecp256k1(qx,qy,pk,d_gtX,d_gtY);

    /* Q1 = u1*G + u2*R (recid=0) */
    uint64_t u2rx[4]={d_u2rx[0],d_u2rx[1],d_u2rx[2],d_u2rx[3]};
    uint64_t u2ry[4]={d_u2ry[0],d_u2ry[1],d_u2ry[2],d_u2ry[3]};
    uint64_t q1x[4],q1y[4],q1z[5];
    memcpy(q1x,qx,32); memcpy(q1y,qy,32);
    q1z[0]=1;q1z[1]=0;q1z[2]=0;q1z[3]=0;q1z[4]=0;
    _PointAddSecp256k1(q1x,q1y,q1z,u2rx,u2ry);

    /* Q2 = Q1 + neg_2u2R (recid=1) */
    uint64_t q2x[4],q2y[4],q2z[5];
    memcpy(q2x,q1x,32); memcpy(q2y,q1y,32); memcpy(q2z,q1z,40);
    uint64_t n2rx[4]={d_neg2u2rx[0],d_neg2u2rx[1],d_neg2u2rx[2],d_neg2u2rx[3]};
    uint64_t n2ry[4]={d_neg2u2ry[0],d_neg2u2ry[1],d_neg2u2ry[2],d_neg2u2ry[3]};
    _PointAddSecp256k1(q2x,q2y,q2z,n2rx,n2ry);

    /* Batch ModInv */
    uint64_t prod[5]={0,0,0,0,0};
    _ModMult(prod,q1z,q2z); _ModInv(prod);
    uint64_t inv1[5],inv2[5];
    _ModMult(inv1,prod,q2z); _ModMult(inv2,prod,q1z);
    _ModMult(q1x,inv1);_ModMult(q1y,inv1);
    _ModMult(q2x,inv2);_ModMult(q2y,inv2);

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
        int vv=easy_mode?gpu_is_der_easy(h,32):(gpu_is_valid_der(h,32) && gpu_der_r_on_curve(h));
        if(vv){v=1;hash_choice=0;recid=ri;break;}
        uint8_t pp[64];memset(pp,0,64);memcpy(pp,h,32);pp[32]=0x80;pp[62]=1;pp[63]=0;
        uint32_t bb2[16];for(int i=0;i<16;i++)bb2[i]=((uint32_t)pp[i*4]<<24)|((uint32_t)pp[i*4+1]<<16)|
            ((uint32_t)pp[i*4+2]<<8)|(uint32_t)pp[i*4+3];
        uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,bb2);
        uint8_t h2[32];for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
            h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
        vv=easy_mode?gpu_is_der_easy(h2,32):(gpu_is_valid_der(h2,32) && gpu_der_r_on_curve(h2));
        if(vv){v=1;hash_choice=1;recid=ri;break;}
    }

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
}

static void compute_gtable(uint8_t *gTableX, uint8_t *gTableY) {
    size_t gt_bytes = 16ULL * 65536 * 32;
    const char *cache = "/tmp/secp256k1_gtable_le.bin";
    FILE *f = fopen(cache, "rb");
    if (f) {
        size_t r1 = fread(gTableX, 1, gt_bytes, f);
        size_t r2 = fread(gTableY, 1, gt_bytes, f);
        fclose(f);
        if (r1 == gt_bytes && r2 == gt_bytes) {
            printf("  GTable loaded from cache\n");
            return;
        }
    }
    printf("  Computing GTable (first run, ~5 min)...\n");
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *x = BN_new(), *y = BN_new(), *shift = BN_new();
    EC_POINT *base = EC_POINT_new(grp), *pt = EC_POINT_new(grp);
    EC_POINT_copy(base, EC_GROUP_get0_generator(grp));
    for (int ch = 0; ch < 16; ch++) {
        if (ch > 0) { BN_set_word(shift, 65536); EC_POINT_mul(grp, base, NULL, base, shift, ctx); }
        EC_POINT_copy(pt, base);
        for (int i = 0; i < 65536; i++) {
            EC_POINT_get_affine_coordinates_GFp(grp, pt, x, y, ctx);
            uint8_t xb[32], yb[32]; memset(xb,0,32); memset(yb,0,32);
            BN_bn2bin(x, xb+(32-BN_num_bytes(x)));
            BN_bn2bin(y, yb+(32-BN_num_bytes(y)));
            /* Convert BE to LE for GPUMath.h */
            for(int j=0;j<16;j++){uint8_t t=xb[j];xb[j]=xb[31-j];xb[31-j]=t;}
            for(int j=0;j<16;j++){uint8_t t=yb[j];yb[j]=yb[31-j];yb[31-j]=t;}
            size_t off = (size_t)ch * 65536 * 32 + (size_t)i * 32;
            memcpy(gTableX + off, xb, 32);
            memcpy(gTableY + off, yb, 32);
            if (i < 65535) EC_POINT_add(grp, pt, pt, base, ctx);
        }
        printf("    Chunk %d/16\n", ch+1);
    }
    BN_free(x); BN_free(y); BN_free(shift);
    EC_POINT_free(base); EC_POINT_free(pt);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
    f = fopen(cache, "wb");
    if (f) { fwrite(gTableX, 1, gt_bytes, f); fwrite(gTableY, 1, gt_bytes, f); fclose(f);
        printf("  GTable saved to cache\n"); }
}

/* Params loader for pinning2.bin */
typedef struct {
    uint32_t midstate[8];
    uint32_t suffix_len;
    uint8_t *suffix;
    uint32_t total_preimage_len;
    uint32_t seq_offset;
    uint32_t lt_offset;
    uint8_t neg_r_inv[32];
    uint8_t u2r_x[32];
    uint8_t u2r_y[32];
} pinning2_params_t;

static int load_pinning2(const char *fn, pinning2_params_t *p) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    if (fread(p->midstate, 4, 8, f) != 8) goto err;
    for (int i=0;i<8;i++) {
        uint8_t *b=(uint8_t*)&p->midstate[i];
        p->midstate[i]=((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    }
    if (fread(&p->suffix_len, 4, 1, f) != 1) goto err;
    p->suffix = (uint8_t*)malloc(p->suffix_len + 16); /* extra for lt+sighash */
    if (fread(p->suffix, 1, p->suffix_len, f) != p->suffix_len) goto err;
    if (fread(&p->total_preimage_len, 4, 1, f) != 1) goto err;
    if (fread(&p->seq_offset, 4, 1, f) != 1) goto err;
    if (fread(&p->lt_offset, 4, 1, f) != 1) goto err;
    if (fread(p->neg_r_inv, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_x, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_y, 1, 32, f) != 32) goto err;
    fclose(f);
    /* Append locktime placeholder (4 bytes) + sighash_type (4 bytes) after suffix */
    /* Locktime goes at lt_offset (= suffix_len), kernel will patch it */
    /* Sighash_type goes right after locktime */
    p->suffix[p->lt_offset] = 0;  /* locktime placeholder (kernel overwrites) */
    p->suffix[p->lt_offset+1] = 0;
    p->suffix[p->lt_offset+2] = 0;
    p->suffix[p->lt_offset+3] = 0;
    p->suffix[p->lt_offset+4] = 0x01; /* SIGHASH_ALL */
    p->suffix[p->lt_offset+5] = 0;
    p->suffix[p->lt_offset+6] = 0;
    p->suffix[p->lt_offset+7] = 0;
    printf("  Loaded: preimage=%u, suffix=%u, seq@%u, lt@%u\n",
           p->total_preimage_len, p->suffix_len, p->seq_offset, p->lt_offset);
    return 0;
err:
    fprintf(stderr, "Error reading %s\n", fn);
    fclose(f); return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <pinning2.bin> [gpu_index] [total_gpus] [global_offset] [easy]\n", argv[0]);
        printf("  total_gpus: total GPUs across ALL machines (default: local count)\n");
        printf("  global_offset: this machine's GPU offset (default: 0)\n");
        return 1;
    }
    int gpu_index = (argc >= 3) ? atoi(argv[2]) : 0;
    int total_gpus_override = (argc >= 4) ? atoi(argv[3]) : 0;
    int global_offset = (argc >= 5) ? atoi(argv[4]) : 0;
    int easy = 0;
    for (int i = 3; i < argc; i++) if (strcmp(argv[i], "easy") == 0) easy = 1;
    
    /* Use the specified GPU */
    cudaSetDevice(gpu_index);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu_index);
    printf("QSB Real Pinning Search (seq+lt) [GPU %d]\n", gpu_index);
    printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    pinning2_params_t pp;
    if (load_pinning2(argv[1], &pp) < 0) return 1;

    /* GTable */
    size_t gt_sz = 16ULL*65536*32;
    uint8_t *h_gtX=(uint8_t*)malloc(gt_sz), *h_gtY=(uint8_t*)malloc(gt_sz);
    compute_gtable(h_gtX, h_gtY);
    uint8_t *d_gtX, *d_gtY;
    cudaMalloc(&d_gtX,gt_sz); cudaMalloc(&d_gtY,gt_sz);
    cudaMemcpy(d_gtX,h_gtX,gt_sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_gtY,h_gtY,gt_sz,cudaMemcpyHostToDevice);
    free(h_gtX); free(h_gtY);

    /* Upload midstate */
    uint32_t *d_mid; cudaMalloc(&d_mid, 32);
    cudaMemcpy(d_mid, pp.midstate, 32, cudaMemcpyHostToDevice);

    /* Build full suffix template: suffix + lt(4) + sighash(4) */
    int full_suffix_len = pp.suffix_len + 8; /* lt + sighash_type already appended partly */
    /* lt_offset in suffix = pp.lt_offset (= pp.suffix_len, right after prefix suffix) */
    /* Actually lt goes at pp.lt_offset which is pp.suffix_len */
    uint8_t *suffix_template = (uint8_t*)calloc(256, 1);
    memcpy(suffix_template, pp.suffix, pp.lt_offset + 8); /* suffix + lt_placeholder + sighash */
    int gpu_suffix_len = pp.suffix_len + 4 + 4; /* suffix + locktime + sighash */

    uint8_t *d_suffix; cudaMalloc(&d_suffix, 256);
    cudaMemcpy(d_suffix, suffix_template, 256, cudaMemcpyHostToDevice);

    printf("  Full suffix: %d bytes, seq@%d, lt@%d\n",
           gpu_suffix_len, pp.seq_offset, pp.lt_offset);
    printf("  Mode: %s\n", easy ? "EASY" : "REAL");

    /* Upload EC constants */
    uint64_t *d_nri, *d_u2rx, *d_u2ry, *d_neg2u2rx, *d_neg2u2ry;
    cudaMalloc(&d_nri,32); cudaMalloc(&d_u2rx,32); cudaMalloc(&d_u2ry,32);
    cudaMalloc(&d_neg2u2rx,32); cudaMalloc(&d_neg2u2ry,32);
    cudaMemcpy(d_nri, pp.neg_r_inv, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2rx, pp.u2r_x, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2ry, pp.u2r_y, 32, cudaMemcpyHostToDevice);

    /* Compute neg_2u2R */
    {
        EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
        BN_CTX *ctx=BN_CTX_new();
        BIGNUM *bx=BN_new(),*by=BN_new();
        uint8_t be[32];
        for(int i=0;i<32;i++) be[i]=pp.u2r_x[31-i]; BN_bin2bn(be,32,bx);
        for(int i=0;i<32;i++) be[i]=pp.u2r_y[31-i]; BN_bin2bn(be,32,by);
        EC_POINT *pt=EC_POINT_new(grp);
        EC_POINT_set_affine_coordinates_GFp(grp,pt,bx,by,ctx);
        EC_POINT *dbl=EC_POINT_new(grp);
        EC_POINT_dbl(grp,dbl,pt,ctx);
        EC_POINT_invert(grp,dbl,ctx);
        BIGNUM *dx=BN_new(),*dy=BN_new();
        EC_POINT_get_affine_coordinates_GFp(grp,dbl,dx,dy,ctx);
        uint8_t dxb[32],dyb[32]; memset(dxb,0,32);memset(dyb,0,32);
        BN_bn2bin(dx,dxb+(32-BN_num_bytes(dx)));
        BN_bn2bin(dy,dyb+(32-BN_num_bytes(dy)));
        uint64_t n2x[4],n2y[4];
        for(int i=0;i<4;i++){n2x[i]=0;n2y[i]=0;
            for(int b=0;b<8;b++){n2x[i]|=(uint64_t)dxb[31-i*8-b]<<(b*8);
                n2y[i]|=(uint64_t)dyb[31-i*8-b]<<(b*8);}}
        cudaMemcpy(d_neg2u2rx,n2x,32,cudaMemcpyHostToDevice);
        cudaMemcpy(d_neg2u2ry,n2y,32,cudaMemcpyHostToDevice);
        BN_free(bx);BN_free(by);BN_free(dx);BN_free(dy);
        EC_POINT_free(pt);EC_POINT_free(dbl);
        EC_GROUP_free(grp);BN_CTX_free(ctx);
    }

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    uint32_t *d_hit_cnt, *d_hit_idx;
    cudaMalloc(&d_hit_cnt,4); cudaMalloc(&d_hit_idx,1024*4);

    int BATCH = 262144;
    int BLKSZ = 256;
    int GRDSZ = (BATCH+BLKSZ-1)/BLKSZ;

    /* Safe ranges */
    uint32_t LT_MIN = 500000000;   /* timestamp interpretation */
    uint32_t LT_MAX = 1744600000;  /* current time (approx) */
    uint32_t SEQ_MIN = 0x80000000; /* bit 31 set — avoids BIP68 */
    uint32_t lt_range = LT_MAX - LT_MIN;

    /* How many GPUs total (for interleaving across all machines) */
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) num_gpus = 1;
    int effective_total = (total_gpus_override > 0) ? total_gpus_override : num_gpus;
    int effective_id = global_offset + gpu_index;

    printf("\n  === Search: lt=[%u,%u] (%u), seq=[0x%08X+], GPU %d (global %d of %d) ===\n",
           LT_MIN, LT_MAX, lt_range, SEQ_MIN, gpu_index, effective_id, effective_total);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t total_searched = 0;
    int found = 0;

    /* Each GPU handles sequences: SEQ_MIN + effective_id, SEQ_MIN + effective_id + effective_total, ... */
    for (uint32_t seq = SEQ_MIN + effective_id; !found; seq += effective_total) {
        /* Search all safe locktimes for this sequence */
        for (uint32_t lt_off = 0; lt_off < lt_range && !found; lt_off += BATCH) {
            uint32_t batch_lt = LT_MIN + lt_off;
            int batch_sz = (lt_off + BATCH <= lt_range) ? BATCH : (lt_range - lt_off);

            uint32_t h_hit = 0;
            cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);

            kernel_pinning_real<<<GRDSZ,BLKSZ>>>(
                d_mid, d_suffix, gpu_suffix_len,
                pp.seq_offset, pp.lt_offset,
                pp.total_preimage_len,
                seq, batch_lt,
                d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
                d_gtX, d_gtY,
                d_hit_cnt, d_hit_idx, batch_sz, easy);
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

            total_searched += batch_sz;

            cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
            if (h_hit > 0) {
                uint32_t hits[64];
                int nh = (h_hit > 64) ? 64 : h_hit;
                cudaMemcpy(hits, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);

                printf("\n  *** HIT! seq=0x%08X ***\n", seq);
                mkdir("results", 0755);
                char fname[256];
                snprintf(fname, sizeof(fname), "results/pinning_hit_%d.txt", gpu_index);
                FILE *f = fopen(fname, "w");
                if (f) {
                    for (int h = 0; h < nh; h++) {
                        uint32_t raw = hits[h];
                        uint32_t lt = batch_lt + (raw & 0x3FFFFFFF);
                        int ri = (raw >> 30) & 1;
                        int hc = (raw >> 31) & 1;
                        fprintf(f, "sequence=%u\nlocktime=%u\nhash_choice=%d\nrecid=%d\n", seq, lt, hc, ri);
                        printf("  seq=0x%08X lt=%u hc=%d recid=%d\n", seq, lt, hc, ri);
                    }
                    fclose(f);
                }
                found = 1;
            }
            
            /* Check if another GPU found it */
            if ((total_searched % (50*1024*1024)) < (uint64_t)BATCH) {
                char check[256];
                for (int g = 0; g < num_gpus; g++) {
                    if (g == gpu_index) continue;
                    snprintf(check, sizeof(check), "results/pinning_hit_%d.txt", g);
                    FILE *cf = fopen(check, "r");
                    if (cf) { fclose(cf); printf("  GPU %d found hit, stopping.\n", g); found = 1; break; }
                }
            }
        }

        /* Progress every 10 sequences */
        uint32_t seqs_done = (seq - SEQ_MIN - effective_id) / effective_total + 1;
        if (seqs_done % 10 == 0 || found) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            double rate = total_searched / elapsed;
            printf("  [GPU %d] seq #%u (0x%08X), %luM total, %.1fM/s, %.0fs\n",
                   gpu_index, seqs_done, seq, total_searched/1000000, rate/1e6, elapsed);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\n  Done: %luM in %.0fs (%.1fM/s), found=%d\n",
           total_searched/1000000, elapsed, total_searched/elapsed/1e6, found);

    return 0;
}
