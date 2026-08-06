/*
 * qsb_search.cu — Production QSB GPU Search
 *
 * Reads binary params from qsb_pipeline.py export.
 * Supports multi-GPU via CUDA_VISIBLE_DEVICES.
 *
 * Build:
 *   nvcc -O3 -maxrregcount=64 -o qsb_search qsb_search.cu -lcrypto -lm
 *
 * Usage:
 *   ./qsb_search pinning pinning.bin [easy]
 *   ./qsb_search digest digest_r1.bin <first_start> <first_end> [easy]
 *   ./qsb_search bench_pinning    # Benchmark with fake data
 *   ./qsb_search bench_digest     # Benchmark with fake data
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

/* GTable */
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

/* DER checks */
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

/* SHA-256 compress */
__device__ void gpu_sha256_compress(uint32_t state[8], const uint32_t block[16]) {
    uint32_t W[64];
    for(int i=0;i<16;i++) W[i]=block[i];
    for(int i=16;i<64;i++){
        uint32_t s0=((W[i-15]>>7)|(W[i-15]<<25))^((W[i-15]>>18)|(W[i-15]<<14))^(W[i-15]>>3);
        uint32_t s1=((W[i-2]>>17)|(W[i-2]<<15))^((W[i-2]>>19)|(W[i-2]<<13))^(W[i-2]>>10);
        W[i]=W[i-16]+s0+W[i-7]+s1;}
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
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

/* SHA-256d from midstate */
__device__ void gpu_sha256d_midstate(uint8_t hash[32], const uint32_t mid[8],
                                      const uint8_t *data, int data_len, int total_len) {
    uint32_t st[8]; for(int i=0;i<8;i++) st[i]=mid[i];
    int off=0;
    while(off+64<=data_len){
        uint32_t blk[16]; for(int i=0;i<16;i++)
            blk[i]=((uint32_t)data[off+i*4]<<24)|((uint32_t)data[off+i*4+1]<<16)|
                   ((uint32_t)data[off+i*4+2]<<8)|(uint32_t)data[off+i*4+3];
        gpu_sha256_compress(st,blk); off+=64;}
    uint8_t pad[128]; memset(pad,0,128);
    int rem=data_len-off; for(int i=0;i<rem;i++) pad[i]=data[off+i]; pad[rem]=0x80;
    int nblk=(rem<56)?1:2;
    uint64_t bits=(uint64_t)total_len*8;
    int last=nblk*64-8;
    for(int i=0;i<8;i++) pad[last+i]=(bits>>(56-i*8))&0xFF;
    for(int b=0;b<nblk;b++){
        uint32_t blk[16]; for(int i=0;i<16;i++)
            blk[i]=((uint32_t)pad[b*64+i*4]<<24)|((uint32_t)pad[b*64+i*4+1]<<16)|
                   ((uint32_t)pad[b*64+i*4+2]<<8)|(uint32_t)pad[b*64+i*4+3];
        gpu_sha256_compress(st,blk);}
    uint8_t first[32];
    for(int i=0;i<8;i++){first[i*4]=(st[i]>>24)&0xFF;first[i*4+1]=(st[i]>>16)&0xFF;
        first[i*4+2]=(st[i]>>8)&0xFF;first[i*4+3]=st[i]&0xFF;}
    uint8_t p2[64]; memset(p2,0,64); for(int i=0;i<32;i++) p2[i]=first[i];
    p2[32]=0x80; p2[62]=0x01; p2[63]=0x00;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t b2[16]; for(int i=0;i<16;i++)
        b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
              ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
    gpu_sha256_compress(s2,b2);
    for(int i=0;i<8;i++){hash[i*4]=(s2[i]>>24)&0xFF;hash[i*4+1]=(s2[i]>>16)&0xFF;
        hash[i*4+2]=(s2[i]>>8)&0xFF;hash[i*4+3]=s2[i]&0xFF;}
}

/* Scalar mul mod N */
__device__ void gpu_scalar_mulmod(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t full[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){unsigned __int128 carry=0;
        for(int j=0;j<4;j++){carry+=(unsigned __int128)full[i+j]+(unsigned __int128)a[i]*b[j];
            full[i+j]=(uint64_t)carry;carry>>=64;}
        full[i+4]+=(uint64_t)carry;}
    const uint64_t RN[4]={0x402DA1732FC9BEBFULL,0x4551231950B75FC4ULL,1ULL,0ULL};
    const uint64_t NN[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
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
                m2[i+j]=(uint64_t)c2;c2>>=64;}}
        carry=0;for(int i=0;i<4;i++){carry+=(unsigned __int128)m2[i]+mid[i];mid[i]=(uint64_t)carry;carry>>=64;}
    }
    r[0]=mid[0];r[1]=mid[1];r[2]=mid[2];r[3]=mid[3];
    for(int rr=0;rr<3;rr++){int ge=0;
        for(int i=3;i>=0;i--){if(r[i]>NN[i]){ge=1;break;}if(r[i]<NN[i])break;if(i==0)ge=1;}
        if(ge){__int128 c=0;for(int i=0;i<4;i++){c+=(__int128)r[i]-NN[i];r[i]=(uint64_t)c;c>>=64;}}}
}

/* EC recovery + SHA-256 double hash + DER check (both recovery flags, batch ModInv) */
__device__ int gpu_ec_recover_check(
    const uint64_t z_le[4], const uint64_t *d_nri,
    const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gtX, uint8_t *d_gtY, int easy_mode,
    int *hash_choice_out, int *recid_out
) {
    uint64_t nri[4]={d_nri[0],d_nri[1],d_nri[2],d_nri[3]};
    uint64_t u1[4]; gpu_scalar_mulmod(u1, nri, z_le);
    uint16_t pk[16]; memcpy(pk, u1, 32);
    uint64_t qx[4],qy[4];
    _PointMultiSecp256k1(qx,qy,pk,d_gtX,d_gtY);

    /* Q1 = u1*G + u2*R (recid=0) */
    uint64_t u2rx[4]={d_u2rx[0],d_u2rx[1],d_u2rx[2],d_u2rx[3]};
    uint64_t u2ry[4]={d_u2ry[0],d_u2ry[1],d_u2ry[2],d_u2ry[3]};
    uint64_t q1x[4],q1y[4],q1z[5];
    memcpy(q1x,qx,32);memcpy(q1y,qy,32);
    q1z[0]=1;q1z[1]=0;q1z[2]=0;q1z[3]=0;q1z[4]=0;
    _PointAddSecp256k1(q1x,q1y,q1z,u2rx,u2ry);

    /* Q2 = Q1 + neg_2u2R = u1*G - u2*R (recid=1) */
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
    uint64_t *pts_x[2]={q1x,q2x};
    uint64_t *pts_y[2]={q1y,q2y};
    for(int ri=0;ri<2;ri++){
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
        int v=easy_mode?gpu_is_der_easy(h,32):gpu_is_valid_der(h,32);
        if(v){*hash_choice_out=0;*recid_out=ri;return 1;}
        uint8_t p2[64];memset(p2,0,64);memcpy(p2,h,32);p2[32]=0x80;p2[62]=1;p2[63]=0;
        uint32_t b2[16];for(int i=0;i<16;i++)b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
            ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
        uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,b2);
        uint8_t h2[32];for(int i=0;i<8;i++){h2[i*4]=(h2s[i]>>24)&0xFF;h2[i*4+1]=(h2s[i]>>16)&0xFF;
            h2[i*4+2]=(h2s[i]>>8)&0xFF;h2[i*4+3]=h2s[i]&0xFF;}
        v=easy_mode?gpu_is_der_easy(h2,32):gpu_is_valid_der(h2,32);
        if(v){*hash_choice_out=1;*recid_out=ri;return 1;}
    }
    return 0;
}

/* ============================================================
 * Pinning kernel (midstate + 2 SHA-256 blocks + EC)
 * ============================================================ */

__global__ void kernel_pinning(
    const uint32_t *d_midstate, const uint8_t *d_tail, int tail_len, int total_len,
    uint32_t start_lt,
    const uint64_t *d_nri, const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gtX, uint8_t *d_gtY,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx, int batch_size, int easy_mode
) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=batch_size) return;
    uint32_t lt = start_lt + (uint32_t)idx;
    
    /* Build final block: tail + locktime(4LE) + sighash_type(4LE) + padding */
    uint8_t buf[64]; memset(buf,0,64);
    for(int i=0;i<tail_len;i++) buf[i]=d_tail[i];
    buf[tail_len]=(lt)&0xFF; buf[tail_len+1]=(lt>>8)&0xFF;
    buf[tail_len+2]=(lt>>16)&0xFF; buf[tail_len+3]=(lt>>24)&0xFF;
    buf[tail_len+4]=0x01; buf[tail_len+5]=0; buf[tail_len+6]=0; buf[tail_len+7]=0;
    int dlen=tail_len+8; buf[dlen]=0x80;
    uint64_t bits=(uint64_t)total_len*8;
    buf[56]=(bits>>56)&0xFF;buf[57]=(bits>>48)&0xFF;buf[58]=(bits>>40)&0xFF;buf[59]=(bits>>32)&0xFF;
    buf[60]=(bits>>24)&0xFF;buf[61]=(bits>>16)&0xFF;buf[62]=(bits>>8)&0xFF;buf[63]=bits&0xFF;
    
    uint32_t blk[16]; for(int i=0;i<16;i++)
        blk[i]=((uint32_t)buf[i*4]<<24)|((uint32_t)buf[i*4+1]<<16)|
               ((uint32_t)buf[i*4+2]<<8)|(uint32_t)buf[i*4+3];
    uint32_t st[8]; for(int i=0;i<8;i++) st[i]=d_midstate[i];
    gpu_sha256_compress(st,blk);
    
    /* Second SHA-256 */
    uint8_t h1[32]; for(int i=0;i<8;i++){h1[i*4]=(st[i]>>24)&0xFF;h1[i*4+1]=(st[i]>>16)&0xFF;
        h1[i*4+2]=(st[i]>>8)&0xFF;h1[i*4+3]=st[i]&0xFF;}
    uint8_t p2[64]; memset(p2,0,64); for(int i=0;i<32;i++) p2[i]=h1[i];
    p2[32]=0x80; p2[62]=0x01; p2[63]=0x00;
    uint32_t b2[16]; for(int i=0;i<16;i++)
        b2[i]=((uint32_t)p2[i*4]<<24)|((uint32_t)p2[i*4+1]<<16)|
              ((uint32_t)p2[i*4+2]<<8)|(uint32_t)p2[i*4+3];
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    gpu_sha256_compress(s2,b2);
    uint8_t sighash[32]; for(int i=0;i<8;i++){sighash[i*4]=(s2[i]>>24)&0xFF;sighash[i*4+1]=(s2[i]>>16)&0xFF;
        sighash[i*4+2]=(s2[i]>>8)&0xFF;sighash[i*4+3]=s2[i]&0xFF;}
    
    /* Convert to LE + EC check */
    uint64_t z[4]; for(int i=0;i<4;i++){z[i]=0;
        for(int b=0;b<8;b++) z[i]|=(uint64_t)sighash[31-i*8-b]<<(b*8);}
    
    int hc=0, ri=0;
    if(gpu_ec_recover_check(z,d_nri,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,d_gtX,d_gtY,easy_mode,&hc,&ri)){
        uint32_t pos=atomicAdd(d_hit_cnt,1);
        if(pos<4096) d_hit_idx[pos]=((uint32_t)idx)|(ri<<30)|(hc<<31);
    }
}

/* ============================================================
 * Digest kernel (build suffix + SHA-256d from midstate + EC)
 * n and t are runtime parameters (passed as kernel args)
 * ============================================================ */

__global__ void kernel_digest(
    const uint8_t *d_combos, int n_pool, int t_sel,
    const uint32_t *d_midstate,
    const uint8_t *d_dummy_sigs, /* n_pool * 10 bytes */
    const uint8_t *d_tail, int tail_len,
    const uint8_t *d_tx_suffix, int tx_suffix_len,
    int total_preimage_len,
    const uint64_t *d_nri, const uint64_t *d_u2rx, const uint64_t *d_u2ry,
    const uint64_t *d_neg2u2rx, const uint64_t *d_neg2u2ry,
    uint8_t *d_gtX, uint8_t *d_gtY,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx, int batch_size, int easy_mode
) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=batch_size) return;
    
    /* Load skip indices */
    uint8_t skip[16]; /* max t=16 */
    for(int i=0;i<t_sel;i++) skip[i]=d_combos[idx*t_sel+i];
    
    /* Build suffix */
    uint8_t suffix[3072]; /* fits n up to ~250 */
    int pos=0;
    int sel=0;
    for(int i=0;i<n_pool;i++){
        if(sel<t_sel && skip[sel]==i){ sel++; }
        else{ for(int b=0;b<10;b++) suffix[pos++]=d_dummy_sigs[i*10+b]; }
    }
    for(int i=0;i<tail_len;i++) suffix[pos++]=d_tail[i];
    for(int i=0;i<tx_suffix_len;i++) suffix[pos++]=d_tx_suffix[i];
    
    /* SHA-256d from midstate */
    uint8_t sighash[32];
    uint32_t mid[8]; for(int i=0;i<8;i++) mid[i]=d_midstate[i];
    gpu_sha256d_midstate(sighash, mid, suffix, pos, total_preimage_len);
    
    /* Convert to LE + EC check */
    uint64_t z[4]; for(int i=0;i<4;i++){z[i]=0;
        for(int b=0;b<8;b++) z[i]|=(uint64_t)sighash[31-i*8-b]<<(b*8);}
    
    int hc2=0, ri2=0;
    if(gpu_ec_recover_check(z,d_nri,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,d_gtX,d_gtY,easy_mode,&hc2,&ri2)){
        uint32_t pos2=atomicAdd(d_hit_cnt,1);
        if(pos2<4096) d_hit_idx[pos2]=((uint32_t)idx)|(ri2<<30)|(hc2<<31);
    }
}

/* ============================================================
 * Host: GTable, param loading, main
 * ============================================================ */

extern "C" {
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
}

#include "qsb_params.h"

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
    if(argc<2){
        printf("Usage:\n");
        printf("  %s pinning <params.bin> [easy]\n", argv[0]);
        printf("  %s digest <params.bin> <first_start> <first_end> [easy]\n", argv[0]);
        return 1;
    }
    
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("QSB GPU Search — %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    
    /* GTable */
    size_t gt_sz=16ULL*65536*32;
    uint8_t *h_gtX=(uint8_t*)malloc(gt_sz),*h_gtY=(uint8_t*)malloc(gt_sz);
    compute_gtable(h_gtX,h_gtY);
    uint8_t *d_gtX,*d_gtY;
    cudaMalloc(&d_gtX,gt_sz);cudaMalloc(&d_gtY,gt_sz);
    cudaMemcpy(d_gtX,h_gtX,gt_sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_gtY,h_gtY,gt_sz,cudaMemcpyHostToDevice);
    free(h_gtX);free(h_gtY);
    
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    uint32_t *d_hit_cnt,*d_hit_idx;
    cudaMalloc(&d_hit_cnt,4);cudaMalloc(&d_hit_idx,4096*4);
    
    int BATCH = 1048576;
    int BLKSZ = 128;
    int GRDSZ = (BATCH+BLKSZ-1)/BLKSZ;
    
    /* ======== PINNING ======== */
    if(strcmp(argv[1],"pinning")==0) {
        if(argc<3){printf("Need params file\n");return 1;}
        int easy = (argc>=4 && strcmp(argv[3],"easy")==0);
        
        pinning_params_t pp;
        if(load_pinning_params(argv[2], &pp)<0) return 1;
        
        /* Upload */
        uint32_t *d_mid; cudaMalloc(&d_mid,32);
        cudaMemcpy(d_mid,pp.midstate,32,cudaMemcpyHostToDevice);
        uint8_t *d_tail; cudaMalloc(&d_tail,pp.tail_data_len);
        cudaMemcpy(d_tail,pp.tail_data,pp.tail_data_len,cudaMemcpyHostToDevice);
        uint64_t *d_nri,*d_u2rx,*d_u2ry,*d_neg2u2rx,*d_neg2u2ry;
        cudaMalloc(&d_nri,32);cudaMalloc(&d_u2rx,32);cudaMalloc(&d_u2ry,32);
        cudaMalloc(&d_neg2u2rx,32);cudaMalloc(&d_neg2u2ry,32);
        cudaMemcpy(d_nri,pp.neg_r_inv,32,cudaMemcpyHostToDevice);
        cudaMemcpy(d_u2rx,pp.u2r_x,32,cudaMemcpyHostToDevice);
        cudaMemcpy(d_u2ry,pp.u2r_y,32,cudaMemcpyHostToDevice);
        
        /* Compute neg_2u2R = -(2 * u2R) for second recovery flag */
        {
            EC_GROUP *grp2=EC_GROUP_new_by_curve_name(NID_secp256k1);
            BN_CTX *ctx2=BN_CTX_new();
            BIGNUM *bx=BN_new(),*by=BN_new();
            /* u2r coords are LE 4×uint64; convert to BE for BN */
            uint8_t be[32];
            for(int i=0;i<32;i++) be[i]=pp.u2r_x[31-i]; BN_bin2bn(be,32,bx);
            for(int i=0;i<32;i++) be[i]=pp.u2r_y[31-i]; BN_bin2bn(be,32,by);
            EC_POINT *pt2=EC_POINT_new(grp2);
            EC_POINT_set_affine_coordinates_GFp(grp2,pt2,bx,by,ctx2);
            EC_POINT *dbl=EC_POINT_new(grp2);
            EC_POINT_dbl(grp2,dbl,pt2,ctx2);
            EC_POINT_invert(grp2,dbl,ctx2);
            BIGNUM *dx=BN_new(),*dy=BN_new();
            EC_POINT_get_affine_coordinates_GFp(grp2,dbl,dx,dy,ctx2);
            uint8_t dx_be[32],dy_be[32]; memset(dx_be,0,32);memset(dy_be,0,32);
            BN_bn2bin(dx,dx_be+(32-BN_num_bytes(dx)));
            BN_bn2bin(dy,dy_be+(32-BN_num_bytes(dy)));
            uint64_t n2x_le[4],n2y_le[4];
            for(int i=0;i<4;i++){n2x_le[i]=0;n2y_le[i]=0;
                for(int b=0;b<8;b++){n2x_le[i]|=(uint64_t)dx_be[31-i*8-b]<<(b*8);
                    n2y_le[i]|=(uint64_t)dy_be[31-i*8-b]<<(b*8);}}
            cudaMemcpy(d_neg2u2rx,n2x_le,32,cudaMemcpyHostToDevice);
            cudaMemcpy(d_neg2u2ry,n2y_le,32,cudaMemcpyHostToDevice);
            BN_free(bx);BN_free(by);BN_free(dx);BN_free(dy);
            EC_POINT_free(pt2);EC_POINT_free(dbl);
            EC_GROUP_free(grp2);BN_CTX_free(ctx2);
        }
        
        printf("  Searching (BATCH=%d, %s)...\n", BATCH, easy?"EASY":"REAL");
        
        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        uint64_t total=0; int found=0;
        
        while(total < (1ULL<<46) && !found) {
            uint32_t h_hit=0; cudaMemcpy(d_hit_cnt,&h_hit,4,cudaMemcpyHostToDevice);
            kernel_pinning<<<GRDSZ,BLKSZ>>>(d_mid,d_tail,pp.tail_data_len,pp.total_preimage_len,
                (uint32_t)total, d_nri,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,d_gtX,d_gtY,
                d_hit_cnt,d_hit_idx,BATCH,easy);
            cudaDeviceSynchronize();
            total+=BATCH;
            
            cudaMemcpy(&h_hit,d_hit_cnt,4,cudaMemcpyDeviceToHost);
            if(h_hit>0){
                uint32_t idxs[16]; cudaMemcpy(idxs,d_hit_idx,64,cudaMemcpyDeviceToHost);
                printf("  HIT! %u found\n",h_hit);
                /* Write results */
                mkdir("results", 0755);
                char fname[256]; snprintf(fname,256,"results/pinning_hit.txt");
                FILE *f=fopen(fname,"w");
                for(uint32_t i=0;i<h_hit&&i<16;i++){
                    uint32_t lt=(uint32_t)(total-BATCH)+idxs[i];
                    fprintf(f,"locktime=%u\n",lt);
                    printf("  locktime=%u\n",lt);
                }
                fclose(f);
                found=1;
            }
            
            if(total%(BATCH*40)==0){
                clock_gettime(CLOCK_MONOTONIC,&t1);
                double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
                printf("  %luM searched, %.0fM/s\n",total/1000000,(double)total/el/1e6);
            }
        }
        
        free_pinning_params(&pp);
        cudaFree(d_mid);cudaFree(d_tail);cudaFree(d_nri);cudaFree(d_u2rx);cudaFree(d_u2ry);
    }
    
    /* ======== DIGEST ======== */
    else if(strcmp(argv[1],"digest")==0) {
        if(argc<5){printf("Need: digest <params.bin> <first_start> <first_end> [easy]\n");return 1;}
        int first_start = atoi(argv[3]);
        int first_end = atoi(argv[4]);
        int easy = (argc>=6 && strcmp(argv[5],"easy")==0);
        
        digest_params_t dp;
        if(load_digest_params(argv[2], &dp)<0) return 1;
        
        int n_pool = dp.n;
        int t_sel = dp.t;
        
        /* Upload */
        uint32_t *d_mid; cudaMalloc(&d_mid,32);
        cudaMemcpy(d_mid,dp.midstate,32,cudaMemcpyHostToDevice);
        uint8_t *d_dsigs; cudaMalloc(&d_dsigs,n_pool*10);
        cudaMemcpy(d_dsigs,dp.dummy_sigs,n_pool*10,cudaMemcpyHostToDevice);
        uint8_t *d_tail; cudaMalloc(&d_tail,dp.tail_section_len);
        cudaMemcpy(d_tail,dp.tail_section,dp.tail_section_len,cudaMemcpyHostToDevice);
        uint8_t *d_suf; cudaMalloc(&d_suf,dp.tx_suffix_len);
        cudaMemcpy(d_suf,dp.tx_suffix,dp.tx_suffix_len,cudaMemcpyHostToDevice);
        uint64_t *d_nri,*d_u2rx,*d_u2ry,*d_neg2u2rx,*d_neg2u2ry;
        cudaMalloc(&d_nri,32);cudaMalloc(&d_u2rx,32);cudaMalloc(&d_u2ry,32);
        cudaMalloc(&d_neg2u2rx,32);cudaMalloc(&d_neg2u2ry,32);
        cudaMemcpy(d_nri,dp.neg_r_inv,32,cudaMemcpyHostToDevice);
        cudaMemcpy(d_u2rx,dp.u2r_x,32,cudaMemcpyHostToDevice);
        cudaMemcpy(d_u2ry,dp.u2r_y,32,cudaMemcpyHostToDevice);
        
        /* Compute neg_2u2R from u2R */
        {
            EC_GROUP *grp2=EC_GROUP_new_by_curve_name(NID_secp256k1);
            BN_CTX *ctx2=BN_CTX_new();
            BIGNUM *bx=BN_new(),*by=BN_new();
            uint8_t be[32];
            for(int i=0;i<32;i++) be[i]=dp.u2r_x[31-i]; BN_bin2bn(be,32,bx);
            for(int i=0;i<32;i++) be[i]=dp.u2r_y[31-i]; BN_bin2bn(be,32,by);
            EC_POINT *pt2=EC_POINT_new(grp2);
            EC_POINT_set_affine_coordinates_GFp(grp2,pt2,bx,by,ctx2);
            EC_POINT *dbl=EC_POINT_new(grp2);
            EC_POINT_dbl(grp2,dbl,pt2,ctx2);
            EC_POINT_invert(grp2,dbl,ctx2);
            BIGNUM *dx=BN_new(),*dy=BN_new();
            EC_POINT_get_affine_coordinates_GFp(grp2,dbl,dx,dy,ctx2);
            uint8_t dx_be[32],dy_be[32]; memset(dx_be,0,32);memset(dy_be,0,32);
            BN_bn2bin(dx,dx_be+(32-BN_num_bytes(dx)));
            BN_bn2bin(dy,dy_be+(32-BN_num_bytes(dy)));
            uint64_t n2x_le[4],n2y_le[4];
            for(int i=0;i<4;i++){n2x_le[i]=0;n2y_le[i]=0;
                for(int b=0;b<8;b++){n2x_le[i]|=(uint64_t)dx_be[31-i*8-b]<<(b*8);
                    n2y_le[i]|=(uint64_t)dy_be[31-i*8-b]<<(b*8);}}
            cudaMemcpy(d_neg2u2rx,n2x_le,32,cudaMemcpyHostToDevice);
            cudaMemcpy(d_neg2u2ry,n2y_le,32,cudaMemcpyHostToDevice);
            BN_free(bx);BN_free(by);BN_free(dx);BN_free(dy);
            EC_POINT_free(pt2);EC_POINT_free(dbl);
            EC_GROUP_free(grp2);BN_CTX_free(ctx2);
        }
        
        uint8_t *h_combos=(uint8_t*)malloc(BATCH*t_sel);
        uint8_t *d_combos; cudaMalloc(&d_combos,BATCH*t_sel);
        
        printf("  Digest search: n=%d, t=%d, first=[%d,%d), %s\n",
               n_pool, t_sel, first_start, first_end, easy?"EASY":"REAL");
        
        struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
        uint64_t total_searched=0; int found=0;
        
        for(int first=first_start; first<first_end && !found; first++){
            int rem_pool = n_pool - first - 1;
            int rem_t = t_sel - 1;
            if(rem_pool < rem_t) continue;
            
            int sub[16]; for(int i=0;i<rem_t;i++) sub[i]=first+1+i;
            int batch_pos=0, exhausted=0;
            
            while(!exhausted && !found){
                while(batch_pos<BATCH && !exhausted){
                    h_combos[batch_pos*t_sel]=(uint8_t)first;
                    for(int i=0;i<rem_t;i++) h_combos[batch_pos*t_sel+1+i]=(uint8_t)sub[i];
                    batch_pos++;
                    int i=rem_t-1;
                    while(i>=0 && sub[i]==n_pool-rem_t+i) i--;
                    if(i<0){exhausted=1;break;}
                    sub[i]++; for(int j=i+1;j<rem_t;j++) sub[j]=sub[j-1]+1;
                }
                if(batch_pos==0) break;
                
                cudaMemcpy(d_combos,h_combos,batch_pos*t_sel,cudaMemcpyHostToDevice);
                uint32_t h_hit=0; cudaMemcpy(d_hit_cnt,&h_hit,4,cudaMemcpyHostToDevice);
                int grdsz=(batch_pos+BLKSZ-1)/BLKSZ;
                kernel_digest<<<grdsz,BLKSZ>>>(d_combos,n_pool,t_sel,
                    d_mid,d_dsigs,d_tail,dp.tail_section_len,
                    d_suf,dp.tx_suffix_len,dp.total_preimage_len,
                    d_nri,d_u2rx,d_u2ry,d_neg2u2rx,d_neg2u2ry,d_gtX,d_gtY,
                    d_hit_cnt,d_hit_idx,batch_pos,easy);
                cudaDeviceSynchronize();
                total_searched+=batch_pos;
                batch_pos=0;
                
                cudaMemcpy(&h_hit,d_hit_cnt,4,cudaMemcpyDeviceToHost);
                if(h_hit>0){
                    uint32_t idxs[64]; int nh=(h_hit>64)?64:h_hit;
                    cudaMemcpy(idxs,d_hit_idx,nh*4,cudaMemcpyDeviceToHost);
                    printf("  HIT! %u found at first=%d\n",h_hit,first);
                    mkdir("results",0755);
                    char fname[256]; snprintf(fname,256,"results/digest_hit.txt");
                    FILE *f=fopen(fname,"w");
                    fprintf(f,"first=%d\nhit_count=%u\ntotal_searched=%lu\n",first,h_hit,total_searched);
                    /* To recover exact combo: need to re-enumerate. Store batch context. */
                    for(int h=0;h<nh;h++) fprintf(f,"batch_idx=%u\n",idxs[h]);
                    fclose(f);
                    found=1;
                }
            }
            
            if(total_searched%(10000000/BATCH*BATCH)<(uint64_t)BATCH){
                clock_gettime(CLOCK_MONOTONIC,&t1);
                double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
                printf("  first=%d, %luM, %.1fM/s, %.0fs\n",
                       first,total_searched/1000000,total_searched/el/1e6,el);
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        printf("\n  Done: %luM in %.0fs (%.1fM/s)\n",total_searched/1000000,el,total_searched/el/1e6);
        
        free(h_combos);free_digest_params(&dp);
        cudaFree(d_mid);cudaFree(d_dsigs);cudaFree(d_tail);cudaFree(d_suf);
        cudaFree(d_nri);cudaFree(d_u2rx);cudaFree(d_u2ry);cudaFree(d_combos);
    }
    
    cudaFree(d_gtX);cudaFree(d_gtY);cudaFree(d_hit_cnt);cudaFree(d_hit_idx);
    return 0;
}
