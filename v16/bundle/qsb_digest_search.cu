/* qsb_digest_search.cu — Multi-GPU digest round search
 *
 * Reads digest_rN.bin, enumerates C(130,9) combinations.
 * CPU generates combo batches, GPU hashes + EC recovery + 4 DER checks.
 *
 * Build:  nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm
 * Usage:  ./qsb_digest <digest_rN.bin> <gpu_index> [easy]
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

/* GTable */
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

/* Check if x is a valid x-coordinate on secp256k1 */
__device__ int gpu_is_on_curve(uint64_t *x) {
    uint64_t x2[4], x3[4];
    _ModSqr(x2, x);
    _ModMult(x3, x2, x);
    uint64_t y_sq[4];
    uint64_t c;
    y_sq[0] = x3[0] + 7ULL; c = (y_sq[0] < x3[0]);
    y_sq[1] = x3[1] + c; c = (y_sq[1] < x3[1]);
    y_sq[2] = x3[2] + c; c = (y_sq[2] < x3[2]);
    y_sq[3] = x3[3] + c;
    const uint64_t P_LO = 0xFFFFFFFEFFFFFC2FULL;
    if (y_sq[3] == 0xFFFFFFFFFFFFFFFFULL && y_sq[2] == 0xFFFFFFFFFFFFFFFFULL
        && y_sq[1] == 0xFFFFFFFFFFFFFFFFULL && y_sq[0] >= P_LO) {
        y_sq[0] = y_sq[0] - P_LO;
        y_sq[1] = 0; y_sq[2] = 0; y_sq[3] = 0;
    }
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
    const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,
                         0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    uint64_t rn[4]; uint64_t c;
    uint64_t t = r[0] + N[0]; c = (t < r[0]); rn[0] = t;
    t = r[1] + N[1] + c; c = (t < r[1]) || (c && t == r[1]); rn[1] = t;
    t = r[2] + N[2] + c; c = (t < r[2]) || (c && t == r[2]); rn[2] = t;
    t = r[3] + N[3] + c; if (t < r[3] || (c && t == r[3])) return 0;
    rn[3] = t;
    const uint64_t P_LO = 0xFFFFFFFEFFFFFC2FULL;
    int rn_lt_p = (rn[3] != 0xFFFFFFFFFFFFFFFFULL) ||
                  (rn[2] != 0xFFFFFFFFFFFFFFFFULL) ||
                  (rn[1] != 0xFFFFFFFFFFFFFFFFULL) ||
                  (rn[0] < P_LO);
    if (!rn_lt_p) return 0;
    return gpu_is_on_curve(rn);
}

/* Scalar mulmod */
/* Correct modular multiplication mod secp256k1 group order n
 * Uses reduction: 2^256 ≡ c (mod n) where c = 2^256 - n (129 bits) */
__device__ void gpu_scalar_mulmod(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    /* FIXED v16: incremental carry propagation to avoid 129-bit overflow.
     *
     * The original used __int128 accumulators per limb and summed up to 4
     * 128-bit products into a single slot. For some inputs t[3] (or t[4],
     * t[5]) reaches 129 bits, overflowing __int128 and producing garbage
     * after the carry shift. Verified failing inputs included nri*z for
     * z = a8f2d43a... and z = N-1.
     *
     * Fix: after each product, add it into the limb AND propagate the carry
     * immediately into the next limb. Each limb stays <= 64 bits.
     */
    
    /* Step 1: 512-bit schoolbook multiplication with incremental carry */
    uint64_t p[8] = {0};
    for (int i = 0; i < 4; i++) {
        __uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t v = (__uint128_t)p[i+j] + (__uint128_t)a[i]*b[j] + carry;
            p[i+j] = (uint64_t)v;
            carry = v >> 64;
        }
        p[i+4] = (uint64_t)carry;
    }
    
    /* c = 2^256 - n = {C0, C1, 1, 0} */
    const uint64_t C0=0x402DA1732FC9BEBFULL, C1=0x4551231950B75FC4ULL;
    
    /* Step 2: first reduction — q = p_hi * c + p_lo, with incremental carry */
    uint64_t q[8] = {0};
    for (int i = 0; i < 4; i++) q[i] = p[i];
    /* Add p_hi[i] * C0 starting at q[i] */
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t v = (__uint128_t)q[i] + (__uint128_t)p[4+i] * C0 + carry;
            q[i] = (uint64_t)v;
            carry = v >> 64;
        }
        q[4] = carry;
    }
    /* Add p_hi[i] * C1 starting at q[i+1] */
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t v = (__uint128_t)q[i+1] + (__uint128_t)p[4+i] * C1 + carry;
            q[i+1] = (uint64_t)v;
            carry = v >> 64;
        }
        __uint128_t v = (__uint128_t)q[5] + carry;
        q[5] = (uint64_t)v;
        if (v >> 64) {
            v = (__uint128_t)q[6] + (v >> 64);
            q[6] = (uint64_t)v;
            if (v >> 64) q[7] += (uint64_t)(v >> 64);
        }
    }
    /* Add p_hi[i] * 1 starting at q[i+2] */
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t v = (__uint128_t)q[i+2] + (__uint128_t)p[4+i] + carry;
            q[i+2] = (uint64_t)v;
            carry = v >> 64;
        }
        if (carry) {
            __uint128_t v = (__uint128_t)q[6] + carry;
            q[6] = (uint64_t)v;
            if (v >> 64) q[7] += (uint64_t)(v >> 64);
        }
    }
    
    /* Step 3: second reduction — same as step 2, with q in place of p */
    uint64_t r2[8] = {0};
    for (int i = 0; i < 4; i++) r2[i] = q[i];
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t v = (__uint128_t)r2[i] + (__uint128_t)q[4+i] * C0 + carry;
            r2[i] = (uint64_t)v;
            carry = v >> 64;
        }
        r2[4] = carry;
    }
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t v = (__uint128_t)r2[i+1] + (__uint128_t)q[4+i] * C1 + carry;
            r2[i+1] = (uint64_t)v;
            carry = v >> 64;
        }
        __uint128_t v = (__uint128_t)r2[5] + carry;
        r2[5] = (uint64_t)v;
        if (v >> 64) r2[6] += (uint64_t)(v >> 64);
    }
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t v = (__uint128_t)r2[i+2] + (__uint128_t)q[4+i] + carry;
            r2[i+2] = (uint64_t)v;
            carry = v >> 64;
        }
        if (carry) r2[6] += carry;
    }
    
    uint64_t res[5];
    for (int i = 0; i < 5; i++) res[i] = r2[i];
    
    /* Step 4: conditional subtraction of N (up to 5 reps to be safe) */
    const uint64_t N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
    for (int rep = 0; rep < 5; rep++) {
        int ge = 0;
        if (res[4] > 0) ge = 1;
        else { ge = 1; for (int i = 3; i >= 0; i--) { if (res[i] > N[i]) { ge = 1; break; } if (res[i] < N[i]) { ge = 0; break; } } }
        if (!ge) break;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t old = res[i];
            uint64_t sub = N[i] + borrow;
            res[i] = old - sub;
            borrow = (old < sub || (borrow && N[i] == 0xFFFFFFFFFFFFFFFFULL)) ? 1 : 0;
        }
        if (res[4] > 0) res[4]--;
    }
    
    for (int i = 0; i < 4; i++) r[i] = res[i];
}

/* ============================================================
 * Digest kernel: each thread processes one combination
 * Combo = 9 indices identifying which dummy sigs to SKIP
 * ============================================================ */

#define MAX_N 150
#define MAX_T 16
#define SIG_PUSH_SIZE 10

__global__ void kernel_digest(
    const uint8_t *d_combos,       /* batch × T bytes: indices per combo */
    int n_pool, int t_sel,
    const uint32_t *d_midstate,
    const uint8_t *d_prefix_remainder,
    int prefix_remainder_len,
    const uint8_t *d_dummy_sigs,   /* n_pool × SIG_PUSH_SIZE */
    const uint8_t *d_tail,
    int tail_len,
    const uint8_t *d_tx_suffix,
    int tx_suffix_len,
    int total_preimage_len,
    const uint64_t *d_nri,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gtX, uint8_t *d_gtY,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx,
    uint8_t *d_hit_combos, uint8_t *d_hit_sighash,
    uint8_t *d_hit_keynonce, uint8_t *d_hit_pubhash,
    uint8_t *d_hit_qx, uint8_t *d_hit_qy,
    int batch_size, int easy_mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    /* Load this thread's skip indices */
    uint8_t skip[MAX_T];
    for (int i = 0; i < t_sel; i++)
        skip[i] = d_combos[idx * t_sel + i];

    /* Build suffix: [prefix_remainder] + remaining dummy sigs + tail + tx_suffix */
    uint8_t suffix[8192];
    int pos = 0;
    /* Prepend prefix_remainder bytes (these are the tail end of fixed_prefix
     * that didn't fit into the midstate's full-block boundary). */
    for (int i = 0; i < prefix_remainder_len; i++)
        suffix[pos++] = d_prefix_remainder[i];
    int sel = 0;
    for (int i = 0; i < n_pool; i++) {
        if (sel < t_sel && skip[sel] == i) { sel++; continue; }
        for (int b = 0; b < SIG_PUSH_SIZE; b++)
            suffix[pos++] = d_dummy_sigs[i * SIG_PUSH_SIZE + b];
    }
    for (int i = 0; i < tail_len; i++) suffix[pos++] = d_tail[i];
    for (int i = 0; i < tx_suffix_len; i++) suffix[pos++] = d_tx_suffix[i];

    /* SHA-256 from midstate */
    uint32_t state[8];
    for (int i = 0; i < 8; i++) state[i] = d_midstate[i];

    /* Process full 64-byte blocks */
    int full_blocks = pos / 64;
    for (int b = 0; b < full_blocks; b++) {
        uint32_t blk[16];
        for (int i = 0; i < 16; i++)
            blk[i] = ((uint32_t)suffix[b*64+i*4]<<24)|((uint32_t)suffix[b*64+i*4+1]<<16)|
                     ((uint32_t)suffix[b*64+i*4+2]<<8)|(uint32_t)suffix[b*64+i*4+3];
        _SHA256Transform(state, blk);
    }

    /* Final block with padding */
    uint8_t last_block[128];
    int rem = pos - full_blocks * 64;
    memset(last_block, 0, 128);
    memcpy(last_block, suffix + full_blocks * 64, rem);
    last_block[rem] = 0x80;
    int nblk = (rem < 56) ? 1 : 2;
    uint64_t bit_len = (uint64_t)total_preimage_len * 8;
    int last = nblk * 64 - 8;
    last_block[last]=(bit_len>>56)&0xFF; last_block[last+1]=(bit_len>>48)&0xFF;
    last_block[last+2]=(bit_len>>40)&0xFF; last_block[last+3]=(bit_len>>32)&0xFF;
    last_block[last+4]=(bit_len>>24)&0xFF; last_block[last+5]=(bit_len>>16)&0xFF;
    last_block[last+6]=(bit_len>>8)&0xFF; last_block[last+7]=bit_len&0xFF;

    for (int b = 0; b < nblk; b++) {
        uint32_t blk[16];
        for (int i = 0; i < 16; i++)
            blk[i] = ((uint32_t)last_block[b*64+i*4]<<24)|((uint32_t)last_block[b*64+i*4+1]<<16)|
                     ((uint32_t)last_block[b*64+i*4+2]<<8)|(uint32_t)last_block[b*64+i*4+3];
        _SHA256Transform(state, blk);
    }

    /* Second SHA-256 (SHA-256d) */
    uint8_t first_hash[32];
    for (int i = 0; i < 8; i++) {
        first_hash[i*4]=(state[i]>>24)&0xFF; first_hash[i*4+1]=(state[i]>>16)&0xFF;
        first_hash[i*4+2]=(state[i]>>8)&0xFF; first_hash[i*4+3]=state[i]&0xFF;
    }
    uint8_t p2[64]; memset(p2,0,64); memcpy(p2,first_hash,32);
    p2[32]=0x80; p2[62]=0x01; p2[63]=0x00;
    uint32_t b2[16];
    for (int i=0;i<16;i++) b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
        ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2, b2);

    uint8_t sighash[32];
    for (int i=0;i<8;i++) {
        sighash[i*4]=(s2[i]>>24)&0xFF; sighash[i*4+1]=(s2[i]>>16)&0xFF;
        sighash[i*4+2]=(s2[i]>>8)&0xFF; sighash[i*4+3]=s2[i]&0xFF;
    }

    /* EC recovery with both flags + batch ModInv + 4 DER checks */
    uint64_t z[4]; for(int i=0;i<4;i++){z[i]=0;
        for(int b=0;b<8;b++) z[i]|=(uint64_t)sighash[31-i*8-b]<<(b*8);}
    uint64_t nri[4]={d_nri[0],d_nri[1],d_nri[2],d_nri[3]};
    uint64_t u1[4]; gpu_scalar_mulmod(u1, nri, z);
    uint16_t pk[16]; memcpy(pk, u1, 32);
    uint64_t qx[4],qy[4];
    _PointMultiSecp256k1(qx,qy,pk,d_gtX,d_gtY);

    uint64_t u2rx[4]={d_u2rx[0],d_u2rx[1],d_u2rx[2],d_u2rx[3]};
    uint64_t u2ry[4]={d_u2ry[0],d_u2ry[1],d_u2ry[2],d_u2ry[3]};
    uint64_t q1x[4],q1y[4],q1z[5];
    memcpy(q1x,qx,32);memcpy(q1y,qy,32);
    q1z[0]=1;q1z[1]=0;q1z[2]=0;q1z[3]=0;q1z[4]=0;
    _PointAddSecp256k1(q1x,q1y,q1z,u2rx,u2ry);

    uint64_t q2x[4],q2y[4],q2z[5];
    memcpy(q2x,q1x,32);memcpy(q2y,q1y,32);memcpy(q2z,q1z,40);
    uint64_t n2rx[4]={d_neg2u2rx[0],d_neg2u2rx[1],d_neg2u2rx[2],d_neg2u2rx[3]};
    uint64_t n2ry[4]={d_neg2u2ry[0],d_neg2u2ry[1],d_neg2u2ry[2],d_neg2u2ry[3]};
    _PointAddSecp256k1(q2x,q2y,q2z,n2rx,n2ry);

    uint64_t prod[5]={0,0,0,0,0};
    _ModMult(prod,q1z,q2z);_ModInv(prod);
    uint64_t inv1[5],inv2[5];
    _ModMult(inv1,prod,q2z);_ModMult(inv2,prod,q1z);
    _ModMult(q1x,inv1);_ModMult(q1y,inv1);
    _ModMult(q2x,inv2);_ModMult(q2y,inv2);

    int v=0, hash_choice=0, recid=0;
    uint64_t *pts_x[2]={q1x,q2x};
    uint64_t *pts_y[2]={q1y,q2y};
    /* Capture the winning intermediates for diagnostics */
    uint8_t winning_pubkey[33] = {0};
    uint8_t winning_hash[32] = {0};
    uint8_t winning_qx[32] = {0};
    uint8_t winning_qy[32] = {0};
    for(int ri=0;ri<2&&!v;ri++){
        uint32_t *x32=(uint32_t*)pts_x[ri];
        uint32_t pb[16];
        uint8_t prefix_byte = 0x2+(uint8_t)(pts_y[ri][0]&1);
        pb[0]=__byte_perm(x32[7],prefix_byte,0x4321);
        pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
        pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
        pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
        pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
        uint8_t h[32];for(int i=0;i<8;i++){h[i*4]=(hs[i]>>24)&0xFF;h[i*4+1]=(hs[i]>>16)&0xFF;
            h[i*4+2]=(hs[i]>>8)&0xFF;h[i*4+3]=hs[i]&0xFF;}
        int vv=easy_mode?gpu_is_der_easy(h,32):(gpu_is_valid_der(h,32) && gpu_der_r_on_curve(h));
        if(vv){
            v=1;hash_choice=0;recid=ri;
            /* Build pubkey: prefix byte + 32 X bytes (BE) */
            winning_pubkey[0] = prefix_byte;
            uint64_t *xp = pts_x[ri];
            uint8_t *xb = (uint8_t*)xp;
            /* X is stored as 4 little-endian uint64s. To get big-endian X bytes:
             * Byte i of BE X = byte (31-i) of LE memory */
            for(int i=0;i<32;i++) winning_pubkey[1+i] = xb[31-i];
            for(int i=0;i<32;i++) winning_hash[i] = h[i];
            uint8_t *yp = (uint8_t*)pts_y[ri];
            for(int i=0;i<32;i++) winning_qx[i] = xb[31-i];
            for(int i=0;i<32;i++) winning_qy[i] = yp[31-i];
            break;
        }
        uint8_t pp[64];memset(pp,0,64);memcpy(pp,h,32);pp[32]=0x80;pp[62]=1;pp[63]=0;
        uint32_t bb2[16];for(int i=0;i<16;i++)bb2[i]=((uint32_t)pp[i*4]<<24)|((uint32_t)pp[i*4+1]<<16)|
            ((uint32_t)pp[i*4+2]<<8)|(uint32_t)pp[i*4+3];
        uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,bb2);
        uint8_t h2[32];for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
            h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
        vv=easy_mode?gpu_is_der_easy(h2,32):(gpu_is_valid_der(h2,32) && gpu_der_r_on_curve(h2));
        if(vv){
            v=1;hash_choice=1;recid=ri;
            winning_pubkey[0] = prefix_byte;
            uint64_t *xp = pts_x[ri];
            uint8_t *xb = (uint8_t*)xp;
            for(int i=0;i<32;i++) winning_pubkey[1+i] = xb[31-i];
            for(int i=0;i<32;i++) winning_hash[i] = h2[i];
            uint8_t *yp = (uint8_t*)pts_y[ri];
            for(int i=0;i<32;i++) winning_qx[i] = xb[31-i];
            for(int i=0;i<32;i++) winning_qy[i] = yp[31-i];
            break;
        }
    }

    if(v){uint32_t p=atomicAdd(d_hit_cnt,1);
        if(p<1024) {
            d_hit_idx[p]=((uint32_t)idx)|(recid<<30)|(hash_choice<<31);
            for(int i=0;i<t_sel;i++) d_hit_combos[p*MAX_T+i] = skip[i];
            for(int i=0;i<32;i++) d_hit_sighash[p*32+i] = sighash[i];
            /* v14: also store key_nonce (33B), pubhash (32B), qx/qy */
            for(int i=0;i<33;i++) d_hit_keynonce[p*33+i] = winning_pubkey[i];
            for(int i=0;i<32;i++) d_hit_pubhash[p*32+i] = winning_hash[i];
            for(int i=0;i<32;i++) d_hit_qx[p*32+i] = winning_qx[i];
            for(int i=0;i<32;i++) d_hit_qy[p*32+i] = winning_qy[i];
        }
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
}

static void compute_gtable(uint8_t *gTableX, uint8_t *gTableY) {
    size_t gt_bytes = 16ULL * 65536 * 32;
    const char *cache = "/tmp/secp256k1_gtable_le.bin";
    FILE *f = fopen(cache, "rb");
    if (f) {
        size_t r1 = fread(gTableX, 1, gt_bytes, f);
        size_t r2 = fread(gTableY, 1, gt_bytes, f);
        fclose(f);
        if (r1 == gt_bytes && r2 == gt_bytes) { printf("  GTable loaded from cache\n"); return; }
    }
    printf("  Computing GTable (~5 min)...\n");
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
    BN_free(x);BN_free(y);BN_free(shift);
    EC_POINT_free(base);EC_POINT_free(pt);
    EC_GROUP_free(grp);BN_CTX_free(ctx);
    f = fopen(cache, "wb");
    if (f) { fwrite(gTableX,1,gt_bytes,f); fwrite(gTableY,1,gt_bytes,f); fclose(f);
        printf("  GTable saved to cache\n"); }
}

/* Digest params loader */
typedef struct {
    uint32_t n, t;
    uint32_t total_preimage_len;
    uint32_t tail_section_len;
    uint32_t tx_suffix_len;
    uint32_t prefix_remainder_len;   /* NEW: bytes of fixed_prefix not in midstate */
    uint32_t midstate[8];
    uint8_t *prefix_remainder;       /* NEW: the up-to-63 bytes before dummy sigs */
    uint8_t *dummy_sigs;
    uint8_t *tail_section;
    uint8_t *tx_suffix;
    uint8_t neg_r_inv[32];
    uint8_t u2r_x[32];
    uint8_t u2r_y[32];
} digest_params_t;

static int load_digest_params(const char *fn, digest_params_t *p) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    if (fread(&p->n, 4, 1, f) != 1) goto err;
    if (fread(&p->t, 4, 1, f) != 1) goto err;
    if (fread(&p->total_preimage_len, 4, 1, f) != 1) goto err;
    if (fread(&p->tail_section_len, 4, 1, f) != 1) goto err;
    if (fread(&p->tx_suffix_len, 4, 1, f) != 1) goto err;
    if (fread(&p->prefix_remainder_len, 4, 1, f) != 1) goto err;
    if (fread(p->midstate, 4, 8, f) != 8) goto err;
    for (int i=0;i<8;i++){
        uint8_t *b=(uint8_t*)&p->midstate[i];
        p->midstate[i]=((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    }
    if (p->prefix_remainder_len > 0) {
        p->prefix_remainder = (uint8_t*)malloc(p->prefix_remainder_len);
        if (fread(p->prefix_remainder, 1, p->prefix_remainder_len, f) != p->prefix_remainder_len) goto err;
    } else {
        p->prefix_remainder = NULL;
    }
    p->dummy_sigs = (uint8_t*)malloc(p->n * SIG_PUSH_SIZE);
    if (fread(p->dummy_sigs, 1, p->n * SIG_PUSH_SIZE, f) != p->n * SIG_PUSH_SIZE) goto err;
    p->tail_section = (uint8_t*)malloc(p->tail_section_len);
    if (fread(p->tail_section, 1, p->tail_section_len, f) != p->tail_section_len) goto err;
    p->tx_suffix = (uint8_t*)malloc(p->tx_suffix_len);
    if (fread(p->tx_suffix, 1, p->tx_suffix_len, f) != p->tx_suffix_len) goto err;
    if (fread(p->neg_r_inv, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_x, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_y, 1, 32, f) != 32) goto err;
    fclose(f);
    printf("  Loaded: n=%u, t=%u, preimage=%u, tail=%u, suffix=%u, prefix_rem=%u\n",
           p->n, p->t, p->total_preimage_len, p->tail_section_len, p->tx_suffix_len,
           p->prefix_remainder_len);
    return 0;
err:
    fprintf(stderr, "Error reading %s\n", fn); fclose(f); return -1;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <digest_rN.bin> <gpu_index> <sequence> <locktime> [total_gpus] [global_offset] [easy]\n", argv[0]);
        printf("  total_gpus: total GPUs across ALL machines (default: local count)\n");
        printf("  global_offset: this machine's GPU offset (default: 0)\n");
        return 1;
    }
    int gpu_index = atoi(argv[2]);
    uint32_t seq_val = (uint32_t)strtoul(argv[3], NULL, 0);
    uint32_t lt_val = (uint32_t)strtoul(argv[4], NULL, 0);
    int total_gpus_override = (argc >= 6) ? atoi(argv[5]) : 0;
    int global_offset = (argc >= 7) ? atoi(argv[6]) : 0;
    int easy = 0;
    for (int i = 5; i < argc; i++) if (strcmp(argv[i], "easy") == 0) easy = 1;

    cudaSetDevice(gpu_index);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, gpu_index);
    printf("QSB Digest Search [GPU %d]\n", gpu_index);
    printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    digest_params_t dp;
    if (load_digest_params(argv[1], &dp) < 0) return 1;

    /* Patch tx_suffix with actual sequence and locktime */
    /* Layout: [seq(4)] [varint(0)(1)] [locktime(4)] [sighash(4)] */
    if (dp.tx_suffix_len >= 13) {
        dp.tx_suffix[0] = seq_val & 0xFF;
        dp.tx_suffix[1] = (seq_val >> 8) & 0xFF;
        dp.tx_suffix[2] = (seq_val >> 16) & 0xFF;
        dp.tx_suffix[3] = (seq_val >> 24) & 0xFF;
        dp.tx_suffix[5] = lt_val & 0xFF;
        dp.tx_suffix[6] = (lt_val >> 8) & 0xFF;
        dp.tx_suffix[7] = (lt_val >> 16) & 0xFF;
        dp.tx_suffix[8] = (lt_val >> 24) & 0xFF;
        printf("  Patched tx_suffix: seq=0x%08X lt=%u\n", seq_val, lt_val);
    }
    
    /* Also fix total_preimage_len if tx_suffix changed size */
    /* (it shouldn't — same 13 bytes either way) */

    int n_pool = dp.n;
    int t_sel = dp.t;

    /* GTable */
    size_t gt_sz = 16ULL*65536*32;
    uint8_t *h_gtX=(uint8_t*)malloc(gt_sz), *h_gtY=(uint8_t*)malloc(gt_sz);
    compute_gtable(h_gtX, h_gtY);
    uint8_t *d_gtX, *d_gtY;
    cudaMalloc(&d_gtX,gt_sz);cudaMalloc(&d_gtY,gt_sz);
    cudaMemcpy(d_gtX,h_gtX,gt_sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_gtY,h_gtY,gt_sz,cudaMemcpyHostToDevice);
    free(h_gtX);free(h_gtY);

    /* Upload params */
    uint32_t *d_mid; cudaMalloc(&d_mid,32);
    cudaMemcpy(d_mid, dp.midstate, 32, cudaMemcpyHostToDevice);
    uint8_t *d_prem = NULL;
    if (dp.prefix_remainder_len > 0) {
        cudaMalloc(&d_prem, dp.prefix_remainder_len);
        cudaMemcpy(d_prem, dp.prefix_remainder, dp.prefix_remainder_len, cudaMemcpyHostToDevice);
    }
    uint8_t *d_dsigs; cudaMalloc(&d_dsigs, n_pool*SIG_PUSH_SIZE);
    cudaMemcpy(d_dsigs, dp.dummy_sigs, n_pool*SIG_PUSH_SIZE, cudaMemcpyHostToDevice);
    uint8_t *d_tail; cudaMalloc(&d_tail, dp.tail_section_len);
    cudaMemcpy(d_tail, dp.tail_section, dp.tail_section_len, cudaMemcpyHostToDevice);
    uint8_t *d_suf; cudaMalloc(&d_suf, dp.tx_suffix_len);
    cudaMemcpy(d_suf, dp.tx_suffix, dp.tx_suffix_len, cudaMemcpyHostToDevice);

    uint64_t *d_nri,*d_u2rx,*d_u2ry,*d_neg2u2rx,*d_neg2u2ry;
    cudaMalloc(&d_nri,32);cudaMalloc(&d_u2rx,32);cudaMalloc(&d_u2ry,32);
    cudaMalloc(&d_neg2u2rx,32);cudaMalloc(&d_neg2u2ry,32);
    cudaMemcpy(d_nri,dp.neg_r_inv,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2rx,dp.u2r_x,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2ry,dp.u2r_y,32,cudaMemcpyHostToDevice);

    /* Compute neg_2u2R */
    {
        EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
        BN_CTX *ctx=BN_CTX_new();
        BIGNUM *bx=BN_new(),*by=BN_new();
        uint8_t be[32];
        for(int i=0;i<32;i++) be[i]=dp.u2r_x[31-i]; BN_bin2bn(be,32,bx);
        for(int i=0;i<32;i++) be[i]=dp.u2r_y[31-i]; BN_bin2bn(be,32,by);
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
    uint8_t *d_hit_combos, *d_hit_sighash;
    uint8_t *d_hit_keynonce, *d_hit_pubhash, *d_hit_qx, *d_hit_qy;
    cudaMalloc(&d_hit_cnt,4);cudaMalloc(&d_hit_idx,1024*4);
    cudaMalloc(&d_hit_combos, 1024 * MAX_T);
    cudaMalloc(&d_hit_sighash, 1024 * 32);
    cudaMalloc(&d_hit_keynonce, 1024 * 33);
    cudaMalloc(&d_hit_pubhash, 1024 * 32);
    cudaMalloc(&d_hit_qx, 1024 * 32);
    cudaMalloc(&d_hit_qy, 1024 * 32);

    int BATCH = 65536;  /* smaller batch — each thread does more work */
    int BLKSZ = 128;
    int GRDSZ = (BATCH+BLKSZ-1)/BLKSZ;

    /* Multi-GPU: each GPU handles every Nth first-index */
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) num_gpus = 1;
    
    /* Support multi-machine: override total GPU count and offset */
    int effective_total = (total_gpus_override > 0) ? total_gpus_override : num_gpus;
    int effective_id = global_offset + gpu_index;

    printf("  Mode: %s, GPU %d (global %d of %d)\n", easy?"EASY":"REAL", gpu_index, effective_id, effective_total);
    printf("  Batch: %d combos per kernel launch\n", BATCH);

    uint8_t *h_combos = (uint8_t*)malloc(BATCH * t_sel);
    uint8_t *d_combos; cudaMalloc(&d_combos, BATCH * t_sel);

    struct timespec t0, t1, t_last_report;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    t_last_report = t0;
    uint64_t total_searched = 0;

    /* Precompute my slice total for progress reporting.
     * Each "first" index contributes C(n_pool - first - 1, t_sel - 1) combos.
     * This GPU handles first = effective_id, effective_id + effective_total, ... */
    auto binom = [](int n, int k) -> uint64_t {
        if (k < 0 || k > n || n < 0) return 0;
        if (k > n - k) k = n - k;
        uint64_t r = 1;
        for (int i = 0; i < k; i++) {
            r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
        }
        return r;
    };
    uint64_t my_slice_total = 0;
    for (int f = effective_id; f <= n_pool - t_sel; f += effective_total) {
        my_slice_total += binom(n_pool - f - 1, t_sel - 1);
    }
    /* Also compute the total across ALL GPUs for context */
    uint64_t global_total = binom(n_pool, t_sel);
    printf("  Search space (GLOBAL): C(%d,%d) = %llu combos\n",
           n_pool, t_sel, (unsigned long long)global_total);
    printf("  Search space (this GPU's slice): %llu combos (%.3f%% of global)\n",
           (unsigned long long)my_slice_total,
           100.0 * my_slice_total / (double)global_total);
    int found = 0;

    /* Enumerate combos: outer loop by first index, interleaved across GPUs */
    for (int first = effective_id; first <= n_pool - t_sel && !found; first += effective_total) {
        /* For this first index, enumerate all C(n-first-1, t-1) remaining combos */
        int sub[MAX_T];
        for (int i = 0; i < t_sel - 1; i++) sub[i] = first + 1 + i;
        int batch_pos = 0;
        int exhausted = 0;

        while (!exhausted && !found) {
            /* Fill batch */
            while (batch_pos < BATCH && !exhausted) {
                h_combos[batch_pos * t_sel] = (uint8_t)first;
                for (int i = 0; i < t_sel - 1; i++)
                    h_combos[batch_pos * t_sel + 1 + i] = (uint8_t)sub[i];
                batch_pos++;

                /* Next combo (lexicographic) */
                int i = t_sel - 2;
                while (i >= 0 && sub[i] == n_pool - (t_sel - 1) + i) i--;
                if (i < 0) { exhausted = 1; break; }
                sub[i]++;
                for (int j = i + 1; j < t_sel - 1; j++) sub[j] = sub[j-1] + 1;
            }
            if (batch_pos == 0) break;

            /* Upload and run */
            cudaMemcpy(d_combos, h_combos, batch_pos * t_sel, cudaMemcpyHostToDevice);
            uint32_t h_hit = 0;
            cudaMemcpy(d_hit_cnt, &h_hit, 4, cudaMemcpyHostToDevice);

            int grdsz = (batch_pos + BLKSZ - 1) / BLKSZ;
            kernel_digest<<<grdsz, BLKSZ>>>(
                d_combos, n_pool, t_sel,
                d_mid,
                d_prem, (int)dp.prefix_remainder_len,
                d_dsigs, d_tail, dp.tail_section_len,
                d_suf, dp.tx_suffix_len, dp.total_preimage_len,
                d_nri, d_u2rx, d_u2ry, d_neg2u2rx, d_neg2u2ry,
                d_gtX, d_gtY,
                d_hit_cnt, d_hit_idx,
                d_hit_combos, d_hit_sighash,
                d_hit_keynonce, d_hit_pubhash,
                d_hit_qx, d_hit_qy,
                batch_pos, easy);
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

            total_searched += batch_pos;
            batch_pos = 0;

            cudaMemcpy(&h_hit, d_hit_cnt, 4, cudaMemcpyDeviceToHost);
            if (h_hit > 0) {
                uint32_t hits[64];
                int nh = (h_hit > 64) ? 64 : h_hit;
                cudaMemcpy(hits, d_hit_idx, nh*4, cudaMemcpyDeviceToHost);

                printf("\n  *** DIGEST HIT! ***\n");
                mkdir("results", 0755);
                char fname[256];
                snprintf(fname, sizeof(fname), "results/digest_hit_%d.txt", gpu_index);
                FILE *ff = fopen(fname, "w");
                if (ff) {
                    uint8_t all_combos[1024 * MAX_T];
                    uint8_t all_sighash[1024 * 32];
                    uint8_t all_keynonce[1024 * 33];
                    uint8_t all_pubhash[1024 * 32];
                    uint8_t all_qx[1024 * 32];
                    uint8_t all_qy[1024 * 32];
                    cudaMemcpy(all_combos, d_hit_combos, nh * MAX_T, cudaMemcpyDeviceToHost);
                    cudaMemcpy(all_sighash, d_hit_sighash, nh * 32, cudaMemcpyDeviceToHost);
                    cudaMemcpy(all_keynonce, d_hit_keynonce, nh * 33, cudaMemcpyDeviceToHost);
                    cudaMemcpy(all_pubhash, d_hit_pubhash, nh * 32, cudaMemcpyDeviceToHost);
                    cudaMemcpy(all_qx, d_hit_qx, nh * 32, cudaMemcpyDeviceToHost);
                    cudaMemcpy(all_qy, d_hit_qy, nh * 32, cudaMemcpyDeviceToHost);

                    for (int h = 0; h < nh; h++) {
                        uint32_t raw = hits[h];
                        int combo_idx = raw & 0x3FFFFFFF;
                        int ri = (raw >> 30) & 1;
                        int hc = (raw >> 31) & 1;
                        uint8_t *combo = all_combos + h * MAX_T;
                        uint8_t *sighash_z = all_sighash + h * 32;
                        uint8_t *kn = all_keynonce + h * 33;
                        uint8_t *ph = all_pubhash + h * 32;
                        uint8_t *qx = all_qx + h * 32;
                        uint8_t *qy = all_qy + h * 32;
                        fprintf(ff, "indices=");
                        printf("  indices=");
                        for (int j = 0; j < t_sel; j++) {
                            fprintf(ff, "%s%d", j?",":"", combo[j]);
                            printf("%s%d", j?",":"", combo[j]);
                        }
                        fprintf(ff, "\nhash_choice=%d\nrecid=%d\n", hc, ri);
                        fprintf(ff, "sighash=");
                        for (int j = 0; j < 32; j++) fprintf(ff, "%02x", sighash_z[j]);
                        fprintf(ff, "\ncombo_idx=%d\n", combo_idx);
                        fprintf(ff, "key_nonce=");
                        for (int j = 0; j < 33; j++) fprintf(ff, "%02x", kn[j]);
                        fprintf(ff, "\npubhash=");
                        for (int j = 0; j < 32; j++) fprintf(ff, "%02x", ph[j]);
                        fprintf(ff, "\nqx=");
                        for (int j = 0; j < 32; j++) fprintf(ff, "%02x", qx[j]);
                        fprintf(ff, "\nqy=");
                        for (int j = 0; j < 32; j++) fprintf(ff, "%02x", qy[j]);
                        fprintf(ff, "\n");
                        printf(" hc=%d recid=%d\n", hc, ri);
                        printf("  sighash=");
                        for (int j = 0; j < 32; j++) printf("%02x", sighash_z[j]);
                        printf("\n  key_nonce=");
                        for (int j = 0; j < 33; j++) printf("%02x", kn[j]);
                        printf("\n  pubhash=");
                        for (int j = 0; j < 32; j++) printf("%02x", ph[j]);
                        printf("\n  qx=");
                        for (int j = 0; j < 32; j++) printf("%02x", qx[j]);
                        printf("\n  qy=");
                        for (int j = 0; j < 32; j++) printf("%02x", qy[j]);
                        printf("\n  combo_idx=%d\n", combo_idx);
                    }
                    fclose(ff);
                }
                found = 1;
            }

            /* Check if another GPU found it */
            if ((total_searched % 1000000) < (uint64_t)BATCH) {
                for (int g = 0; g < num_gpus; g++) {
                    if (g == gpu_index) continue;
                    char check[256];
                    snprintf(check, sizeof(check), "results/digest_hit_%d.txt", g);
                    FILE *cf = fopen(check, "r");
                    if (cf) { fclose(cf); printf("  GPU %d found hit\n", g); found = 1; break; }
                }
            }

            /* Periodic progress: print every ~60 seconds rather than only
             * at first-boundary. For t=9 with first=0 that boundary is
             * hours away, so without this we'd never see progress. */
            {
                struct timespec t_now;
                clock_gettime(CLOCK_MONOTONIC, &t_now);
                double secs_since_report = (t_now.tv_sec - t_last_report.tv_sec)
                    + (t_now.tv_nsec - t_last_report.tv_nsec) / 1e9;
                if (secs_since_report >= 60.0) {
                    double elapsed_total = (t_now.tv_sec - t0.tv_sec)
                        + (t_now.tv_nsec - t0.tv_nsec) / 1e9;
                    double rate = total_searched / elapsed_total;
                    double pct = (my_slice_total > 0) ? 100.0 * total_searched / (double)my_slice_total : 0.0;
                    double remaining_sec = (rate > 0 && my_slice_total > total_searched)
                        ? (double)(my_slice_total - total_searched) / rate : 0.0;
                    int eta_h = (int)(remaining_sec / 3600);
                    int eta_m = (int)((remaining_sec - eta_h*3600) / 60);
                    printf("  [GPU %d] first=%d/%d  %.4f%% (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs  ETA=%dh%02dm\n",
                           gpu_index, first, n_pool - t_sel,
                           pct,
                           (unsigned long long)(total_searched/1000000),
                           (unsigned long long)(my_slice_total/1000000),
                           rate/1e6, elapsed_total, eta_h, eta_m);
                    fflush(stdout);
                    t_last_report = t_now;
                }
            }
        }

        /* Progress (end of first value) */
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        {
            double rate = total_searched / elapsed;
            double pct = (my_slice_total > 0) ? 100.0 * total_searched / (double)my_slice_total : 0.0;
            double remaining_sec = (rate > 0 && my_slice_total > total_searched)
                ? (double)(my_slice_total - total_searched) / rate : 0.0;
            int eta_h = (int)(remaining_sec / 3600);
            int eta_m = (int)((remaining_sec - eta_h*3600) / 60);
            printf("  [GPU %d] first=%d/%d DONE  %.2f%% (%lluM/%lluM)  %.1fM/s  elapsed=%.0fs  ETA=%dh%02dm\n",
                   gpu_index, first, n_pool - t_sel,
                   pct,
                   (unsigned long long)(total_searched/1000000),
                   (unsigned long long)(my_slice_total/1000000),
                   rate/1e6, elapsed, eta_h, eta_m);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\n  [GPU %d] Done: %lluM (of %lluM slice) in %.0fs (%.1fM/s), found=%d\n",
           gpu_index,
           (unsigned long long)(total_searched/1000000),
           (unsigned long long)(my_slice_total/1000000),
           elapsed, total_searched/elapsed/1e6, found);

    free(h_combos);
    return 0;
}
