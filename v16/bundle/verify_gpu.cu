/* verify_gpu.cu — Verify GPU sighash + EC recovery matches CPU
 * Compile: nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm
 * Run:     ./verify_gpu pinning2.bin <seq> <lt>
 * Example: ./verify_gpu pinning2.bin 2281702028 600548049
 *
 * Prints intermediate values so we can compare with Python.
 * Runs ONE thread on CPU-side to avoid any GPU weirdness, 
 * then runs ONE thread on GPU, and compares.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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

/* ---- GPU code ---- */

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

__device__ void gpu_scalar_mulmod(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    __int128 t[8]={0};
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) t[i+j]+=(__int128)a[i]*b[j];
    for(int i=0;i<7;i++){t[i+1]+=t[i]>>64;t[i]&=0xFFFFFFFFFFFFFFFFULL;}
    uint64_t p[8]; for(int i=0;i<8;i++) p[i]=(uint64_t)t[i];
    const uint64_t C0=0x402DA1732FC9BEBFULL, C1=0x4551231950B75FC4ULL;
    __int128 acc[8]={0};
    for(int i=0;i<4;i++) acc[i]+=(__int128)p[4+i]*C0;
    for(int i=0;i<4;i++) acc[i+1]+=(__int128)p[4+i]*C1;
    for(int i=0;i<4;i++) acc[i+2]+=(__int128)p[4+i];
    for(int i=0;i<4;i++) acc[i]+=p[i];
    for(int i=0;i<7;i++){acc[i+1]+=acc[i]>>64;acc[i]&=0xFFFFFFFFFFFFFFFFULL;}
    uint64_t q[8]; for(int i=0;i<8;i++) q[i]=(uint64_t)acc[i];
    __int128 acc2[8]={0};
    for(int i=0;i<4;i++) acc2[i]+=(__int128)q[4+i]*C0;
    for(int i=0;i<4;i++) acc2[i+1]+=(__int128)q[4+i]*C1;
    for(int i=0;i<4;i++) acc2[i+2]+=(__int128)q[4+i];
    for(int i=0;i<4;i++) acc2[i]+=q[i];
    for(int i=0;i<7;i++){acc2[i+1]+=acc2[i]>>64;acc2[i]&=0xFFFFFFFFFFFFFFFFULL;}
    uint64_t res[5]; for(int i=0;i<5;i++) res[i]=(uint64_t)acc2[i];
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

struct gpu_output {
    uint8_t sighash[32];
    uint64_t u1[4];
    uint64_t u1Gx[4], u1Gy[4];
    uint64_t Q0x[4], Q0y[4];
    uint64_t Q1x[4], Q1y[4];
    uint8_t pubkey0[33], pubkey1[33];
    uint8_t h1_0[32], h2_0[32]; /* SHA256, SHA256² for recid=0 */
    uint8_t h1_1[32], h2_1[32]; /* SHA256, SHA256² for recid=1 */
};

__global__ void verify_kernel(
    uint32_t *d_midstate, uint8_t *d_suffix, int suffix_len, int total_preimage_len,
    int seq_offset, int lt_offset,
    uint64_t *d_nri, uint64_t *d_u2rx, uint64_t *d_u2ry,
    uint64_t *d_neg2u2rx, uint64_t *d_neg2u2ry,
    uint8_t *d_gtX, uint8_t *d_gtY,
    uint32_t seq_val, uint32_t lt_val,
    struct gpu_output *out)
{
    /* Build suffix with patched seq/lt */
    uint8_t buf[192];
    for(int i=0;i<suffix_len;i++) buf[i]=d_suffix[i];
    buf[seq_offset]=(seq_val)&0xFF; buf[seq_offset+1]=(seq_val>>8)&0xFF;
    buf[seq_offset+2]=(seq_val>>16)&0xFF; buf[seq_offset+3]=(seq_val>>24)&0xFF;
    buf[lt_offset]=(lt_val)&0xFF; buf[lt_offset+1]=(lt_val>>8)&0xFF;
    buf[lt_offset+2]=(lt_val>>16)&0xFF; buf[lt_offset+3]=(lt_val>>24)&0xFF;

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
    memcpy(out->sighash, sighash, 32);

    /* u1 = neg_r_inv * z mod n */
    uint64_t z[4]; for(int i=0;i<4;i++){z[i]=0;for(int b=0;b<8;b++)z[i]|=(uint64_t)sighash[31-i*8-b]<<(b*8);}
    uint64_t nri[4]={d_nri[0],d_nri[1],d_nri[2],d_nri[3]};
    uint64_t u1[4]; gpu_scalar_mulmod(u1,nri,z);
    memcpy(out->u1, u1, 32);

    /* u1*G */
    uint16_t pk[16]; memcpy(pk,u1,32);
    uint64_t qx[4],qy[4]; _PointMultiSecp256k1(qx,qy,pk,d_gtX,d_gtY);
    memcpy(out->u1Gx, qx, 32); memcpy(out->u1Gy, qy, 32);

    /* Q0 = u1*G + u2R */
    uint64_t u2rx[4]={d_u2rx[0],d_u2rx[1],d_u2rx[2],d_u2rx[3]};
    uint64_t u2ry[4]={d_u2ry[0],d_u2ry[1],d_u2ry[2],d_u2ry[3]};
    uint64_t q0x[4],q0y[4],q0z[5];
    memcpy(q0x,qx,32); memcpy(q0y,qy,32);
    q0z[0]=1;q0z[1]=0;q0z[2]=0;q0z[3]=0;q0z[4]=0;
    _PointAddSecp256k1(q0x,q0y,q0z,u2rx,u2ry);

    /* Q1 = Q0 + neg_2u2R */
    uint64_t q1x[4],q1y[4],q1z[5];
    memcpy(q1x,q0x,32); memcpy(q1y,q0y,32); memcpy(q1z,q0z,40);
    uint64_t n2rx[4]={d_neg2u2rx[0],d_neg2u2rx[1],d_neg2u2rx[2],d_neg2u2rx[3]};
    uint64_t n2ry[4]={d_neg2u2ry[0],d_neg2u2ry[1],d_neg2u2ry[2],d_neg2u2ry[3]};
    _PointAddSecp256k1(q1x,q1y,q1z,n2rx,n2ry);

    /* Batch ModInv */
    uint64_t prod[5]={0,0,0,0,0};
    _ModMult(prod,q0z,q1z); _ModInv(prod);
    uint64_t inv0[5],inv1[5];
    _ModMult(inv0,prod,q1z); _ModMult(inv1,prod,q0z);
    _ModMult(q0x,inv0);_ModMult(q0y,inv0);
    _ModMult(q1x,inv1);_ModMult(q1y,inv1);

    memcpy(out->Q0x, q0x, 32); memcpy(out->Q0y, q0y, 32);
    memcpy(out->Q1x, q1x, 32); memcpy(out->Q1y, q1y, 32);

    /* Compress + hash both Q's */
    uint64_t *pts_x[2]={q0x,q1x};
    uint64_t *pts_y[2]={q0y,q1y};
    uint8_t *pk_out[2]={out->pubkey0, out->pubkey1};
    uint8_t *h1_out[2]={out->h1_0, out->h1_1};
    uint8_t *h2_out[2]={out->h2_0, out->h2_1};

    for(int ri=0;ri<2;ri++){
        uint32_t *x32=(uint32_t*)pts_x[ri];
        uint32_t pb[16];
        pb[0]=__byte_perm(x32[7],0x2+(uint8_t)(pts_y[ri][0]&1),0x4321);
        pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
        pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
        pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
        pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;

        /* Extract pubkey */
        for(int i=0;i<8;i++){uint32_t w=pb[i];
            pk_out[ri][i*4]=(w>>24)&0xFF;pk_out[ri][i*4+1]=(w>>16)&0xFF;
            pk_out[ri][i*4+2]=(w>>8)&0xFF;pk_out[ri][i*4+3]=w&0xFF;}
        pk_out[ri][32]=(pb[8]>>24)&0xFF;

        /* SHA-256 */
        uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
        for(int i=0;i<8;i++){h1_out[ri][i*4]=(hs[i]>>24)&0xFF;h1_out[ri][i*4+1]=(hs[i]>>16)&0xFF;
            h1_out[ri][i*4+2]=(hs[i]>>8)&0xFF;h1_out[ri][i*4+3]=hs[i]&0xFF;}

        /* SHA-256² */
        uint8_t pp[64];memset(pp,0,64);memcpy(pp,h1_out[ri],32);pp[32]=0x80;pp[62]=1;pp[63]=0;
        uint32_t bb2[16];for(int i=0;i<16;i++)bb2[i]=((uint32_t)pp[i*4]<<24)|((uint32_t)pp[i*4+1]<<16)|
            ((uint32_t)pp[i*4+2]<<8)|(uint32_t)pp[i*4+3];
        uint32_t h2s[8];_SHA256Initialize(h2s);_SHA256Transform(h2s,bb2);
        for(int i=0;i<8;i++){h2_out[ri][i*4]=(h2s[i]>>24)&0xFF;h2_out[ri][i*4+1]=(h2s[i]>>16)&0xFF;
            h2_out[ri][i*4+2]=(h2s[i]>>8)&0xFF;h2_out[ri][i*4+3]=h2s[i]&0xFF;}
    }
}

/* ---- Host code ---- */
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
        if (r1 == gt_bytes && r2 == gt_bytes) { printf("GTable loaded from cache\n"); return; }
    }
    printf("Computing GTable (~5 min)...\n");
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
            for(int j=0;j<16;j++){uint8_t t=xb[j];xb[j]=xb[31-j];xb[31-j]=t;}
            for(int j=0;j<16;j++){uint8_t t=yb[j];yb[j]=yb[31-j];yb[31-j]=t;}
            size_t off = (size_t)ch * 65536 * 32 + (size_t)i * 32;
            memcpy(gTableX + off, xb, 32);
            memcpy(gTableY + off, yb, 32);
            if (i < 65535) EC_POINT_add(grp, pt, pt, base, ctx);
        }
        printf("  Chunk %d/16\n", ch+1);
    }
    BN_free(x); BN_free(y); BN_free(shift);
    EC_POINT_free(base); EC_POINT_free(pt);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
    f = fopen(cache, "wb");
    if (f) { fwrite(gTableX,1,gt_bytes,f); fwrite(gTableY,1,gt_bytes,f); fclose(f);
        printf("GTable saved to cache\n"); }
}

static void print_hex(const char *label, const uint8_t *data, int len) {
    printf("%s", label);
    for(int i=0;i<len;i++) printf("%02x", data[i]);
    printf("\n");
}

static void print_u64x4(const char *label, const uint64_t *v) {
    printf("%s%016lx %016lx %016lx %016lx\n", label, v[0], v[1], v[2], v[3]);
}

int main(int argc, char **argv) {
    if (argc < 4) { printf("Usage: %s <pinning2.bin> <seq> <lt>\n", argv[0]); return 1; }

    uint32_t seq_val = (uint32_t)strtoul(argv[2], NULL, 0);
    uint32_t lt_val = (uint32_t)strtoul(argv[3], NULL, 0);

    printf("=== GPU EC Verify: seq=%u (0x%08X) lt=%u ===\n\n", seq_val, seq_val, lt_val);

    /* Load pinning2.bin */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("Cannot open %s\n", argv[1]); return 1; }
    uint32_t midstate[8]; fread(midstate, 4, 8, f);
    for(int i=0;i<8;i++){uint8_t *b=(uint8_t*)&midstate[i];
        midstate[i]=((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];}
    uint32_t suffix_len; fread(&suffix_len, 4, 1, f);
    uint8_t suffix[256]; memset(suffix, 0, 256);
    fread(suffix, 1, suffix_len, f);
    uint32_t total; fread(&total, 4, 1, f);
    uint32_t seq_off; fread(&seq_off, 4, 1, f);
    uint32_t lt_off; fread(&lt_off, 4, 1, f);
    uint8_t nri_bytes[32]; fread(nri_bytes, 1, 32, f);
    uint8_t u2rx_bytes[32]; fread(u2rx_bytes, 1, 32, f);
    uint8_t u2ry_bytes[32]; fread(u2ry_bytes, 1, 32, f);
    fclose(f);

    /* Append locktime placeholder + sighash_type after suffix */
    /* lt goes at lt_off (=suffix_len), sighash right after */
    suffix[lt_off+4] = 0x01; /* SIGHASH_ALL */
    uint32_t gpu_suffix_len = lt_off + 8; /* suffix + lt(4) + sighash(4) */

    printf("suffix_len=%u, gpu_suffix_len=%u, total_preimage=%u, seq_off=%u, lt_off=%u\n",
           suffix_len, gpu_suffix_len, total, seq_off, lt_off);
    
    /* The suffix from pinning2.bin does NOT include lt or sighash.
       But the kernel expects lt_offset within a buffer that has them appended.
       Actually, looking at the real kernel code more carefully:
       The suffix from .bin is: ...stuff...seq(at seq_off)...varint(0)
       Then the kernel does NOT append lt — the lt_offset points to where lt goes.
       
       Hmm, let me just match exactly what the real kernel does. */
    
    /* Revert: remove the appended sighash type, match real kernel exactly */
    suffix_len -= 4;
    
    /* In the real kernel, the buf is built from d_suffix, then patched.
       d_suffix was loaded from the .bin, which has lt_offset pointing to 
       where locktime should go IN the suffix. But suffix from .bin has 
       ...seq_bytes...varint(0) and is suffix_len bytes.
       
       The real kernel patches seq at seq_offset and lt at lt_offset.
       lt_offset could be beyond suffix_len if lt is appended after.
       
       Let me just print suffix bytes to understand: */
    printf("Suffix bytes: ");
    for(int i=0;i<(int)suffix_len && i<70;i++) printf("%02x", suffix[i]);
    printf("...\n");
    printf("Suffix last 10 bytes: ");
    for(int i=suffix_len>10?suffix_len-10:0;i<(int)suffix_len;i++) printf("%02x", suffix[i]);
    printf("\n\n");

    /* OK let's just run the GPU with the EXACT same code as the real kernel
       and print results. The kernel code is copy-pasted above. */
    
    /* GTable */
    size_t gt_sz = 16ULL*65536*32;
    uint8_t *h_gtX=(uint8_t*)malloc(gt_sz), *h_gtY=(uint8_t*)malloc(gt_sz);
    compute_gtable(h_gtX, h_gtY);

    /* Print GTable[0] for verification */
    printf("GTable[0] x (first 16 bytes LE): ");
    for(int i=0;i<16;i++) printf("%02x", h_gtX[i]);
    printf("\n\n");

    /* neg_2u2R computation (same as real code) */
    EC_GROUP *grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *bx=BN_new(),*by=BN_new();
    uint8_t be[32];
    for(int i=0;i<32;i++) be[i]=u2rx_bytes[31-i]; BN_bin2bn(be,32,bx);
    for(int i=0;i<32;i++) be[i]=u2ry_bytes[31-i]; BN_bin2bn(be,32,by);
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
    BN_free(bx);BN_free(by);BN_free(dx);BN_free(dy);
    EC_POINT_free(pt);EC_POINT_free(dbl);
    EC_GROUP_free(grp);BN_CTX_free(ctx);

    /* Upload everything to GPU */
    uint32_t *d_mid; cudaMalloc(&d_mid, 32);
    cudaMemcpy(d_mid, midstate, 32, cudaMemcpyHostToDevice);

    uint8_t *d_suffix; cudaMalloc(&d_suffix, suffix_len+16);
    cudaMemcpy(d_suffix, suffix, gpu_suffix_len, cudaMemcpyHostToDevice);

    uint64_t *d_nri, *d_u2rx, *d_u2ry, *d_n2rx, *d_n2ry;
    cudaMalloc(&d_nri,32); cudaMalloc(&d_u2rx,32); cudaMalloc(&d_u2ry,32);
    cudaMalloc(&d_n2rx,32); cudaMalloc(&d_n2ry,32);
    cudaMemcpy(d_nri, nri_bytes, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2rx, u2rx_bytes, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2ry, u2ry_bytes, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2rx, n2x, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2ry, n2y, 32, cudaMemcpyHostToDevice);

    uint8_t *d_gtX, *d_gtY;
    cudaMalloc(&d_gtX, gt_sz); cudaMalloc(&d_gtY, gt_sz);
    cudaMemcpy(d_gtX, h_gtX, gt_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gtY, h_gtY, gt_sz, cudaMemcpyHostToDevice);

    struct gpu_output *d_out;
    cudaMalloc(&d_out, sizeof(struct gpu_output));

    cudaDeviceSetLimit(cudaLimitStackSize, 32768);

    verify_kernel<<<1,1>>>(d_mid, d_suffix, gpu_suffix_len, total, seq_off, lt_off,
        d_nri, d_u2rx, d_u2ry, d_n2rx, d_n2ry, d_gtX, d_gtY,
        seq_val, lt_val, d_out);
    cudaDeviceSynchronize();

    struct gpu_output out;
    cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);

    printf("=== GPU OUTPUT ===\n");
    print_hex("sighash:          ", out.sighash, 32);
    print_u64x4("u1 (LE u64[4]):   ", out.u1);
    print_u64x4("u1*G.x (LE):      ", out.u1Gx);
    print_u64x4("u1*G.y (LE):      ", out.u1Gy);
    print_u64x4("Q0.x (LE):        ", out.Q0x);
    print_u64x4("Q0.y (LE):        ", out.Q0y);
    print_u64x4("Q1.x (LE):        ", out.Q1x);
    print_u64x4("Q1.y (LE):        ", out.Q1y);
    print_hex("pubkey0 (recid=0): ", out.pubkey0, 33);
    print_hex("pubkey1 (recid=1): ", out.pubkey1, 33);
    print_hex("h1_recid0:         ", out.h1_0, 32);
    print_hex("h2_recid0:         ", out.h2_0, 32);
    print_hex("h1_recid1:         ", out.h1_1, 32);
    print_hex("h2_recid1:         ", out.h2_1, 32);

    free(h_gtX); free(h_gtY);
    return 0;
}
