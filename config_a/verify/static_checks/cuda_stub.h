/* Stub headers to let g++ parse .cu files as C++. */
#ifndef CUDA_STUB_H
#define CUDA_STUB_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define __global__
#define __device__
#define __host__
#define __constant__
#define __shared__
#define __forceinline__ inline
#define __restrict__ restrict
#define __noinline__

typedef struct { unsigned x, y, z; } dim3_;
typedef struct { unsigned x, y, z; } uint3_;
#define dim3 dim3_
#define uint3 uint3_
static uint3_ threadIdx = {0, 0, 0};
static uint3_ blockIdx = {0, 0, 0};
static dim3_ blockDim = {1, 1, 1};
static dim3_ gridDim = {1, 1, 1};
typedef struct { unsigned int x, y, z, w; } uint4;
typedef struct { int x, y, z, w; } int4;

typedef int cudaError_t;
#define cudaSuccess 0
/* Template version: accepts typed pointers and does the cast for us */
template<typename T>
static inline cudaError_t cudaMalloc(T **p, size_t n) { *p = (T *)malloc(n); return 0; }
static inline cudaError_t cudaFree(void *p) { free(p); return 0; }
static inline cudaError_t cudaMemcpy(void *d, const void *s, size_t n, int k) { memcpy(d, s, n); return 0; }
static inline cudaError_t cudaMemset(void *p, int v, size_t n) { memset(p, v, n); return 0; }
static inline cudaError_t cudaSetDevice(int d) { return 0; }
static inline cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return 0; }
static inline cudaError_t cudaDeviceSynchronize(void) { return 0; }
static inline cudaError_t cudaGetLastError(void) { return 0; }
static inline const char *cudaGetErrorString(cudaError_t e) { return "ok"; }
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3

typedef struct { char name[256]; int major, minor; size_t totalGlobalMem; int multiProcessorCount; } cudaDeviceProp;
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int i) { return 0; }

static inline unsigned int atomicAdd(unsigned int *addr, unsigned int val) {
    unsigned int old = *addr; *addr += val; return old;
}
static inline unsigned int atomicMin(unsigned int *addr, unsigned int val) {
    if (val < *addr) *addr = val; return *addr;
}
static inline unsigned int __byte_perm(unsigned int a, unsigned int b, unsigned int sel) {
    return (a ^ b) + sel;
}
static inline unsigned int __brev(unsigned int x) { return x; }
static inline int __ffs(int x) { return x ? __builtin_ffs(x) : 0; }
static inline int __clz(int x) { return x ? __builtin_clz(x) : 32; }
static inline int __clzll(unsigned long long x) { return x ? __builtin_clzll(x) : 64; }
static inline int __popc(unsigned int x) { return __builtin_popcount(x); }

typedef struct BIGNUM_ BIGNUM;
typedef struct BN_CTX_ BN_CTX;
typedef struct EC_GROUP_ EC_GROUP;
typedef struct EC_POINT_ EC_POINT;
#define NID_secp256k1 0
extern "C" {
extern BIGNUM *BN_new(void);
extern void BN_free(BIGNUM *);
extern BN_CTX *BN_CTX_new(void);
extern void BN_CTX_free(BN_CTX *);
extern EC_GROUP *EC_GROUP_new_by_curve_name(int);
extern void EC_GROUP_free(EC_GROUP *);
extern EC_POINT *EC_POINT_new(const EC_GROUP *);
extern void EC_POINT_free(EC_POINT *);
extern const EC_POINT *EC_GROUP_get0_generator(const EC_GROUP *);
extern int EC_POINT_copy(EC_POINT *, const EC_POINT *);
extern int EC_POINT_add(const EC_GROUP *, EC_POINT *, const EC_POINT *, const EC_POINT *, BN_CTX *);
extern int EC_POINT_get_affine_coordinates_GFp(const EC_GROUP *, const EC_POINT *, BIGNUM *, BIGNUM *, BN_CTX *);
extern int EC_POINT_mul(const EC_GROUP *, EC_POINT *, const BIGNUM *, const EC_POINT *, const BIGNUM *, BN_CTX *);
extern int BN_set_word(BIGNUM *, unsigned long);
extern int BN_hex2bn(BIGNUM **, const char *);
extern int BN_bn2bin(const BIGNUM *, unsigned char *);
extern int BN_add(BIGNUM *, const BIGNUM *, const BIGNUM *);
extern int BN_num_bits(const BIGNUM *);
extern int BN_num_bytes(const BIGNUM *);
}
#endif
extern "C" {
extern BIGNUM *BN_bin2bn(const unsigned char *, int, BIGNUM *);
extern int EC_POINT_set_affine_coordinates_GFp(const EC_GROUP *, EC_POINT *, const BIGNUM *, const BIGNUM *, BN_CTX *);
extern int EC_POINT_dbl(const EC_GROUP *, EC_POINT *, const EC_POINT *, BN_CTX *);
extern int EC_POINT_invert(const EC_GROUP *, EC_POINT *, BN_CTX *);
extern const BIGNUM *EC_GROUP_get0_order(const EC_GROUP *);
extern int BN_mod_mul(BIGNUM *, const BIGNUM *, const BIGNUM *, const BIGNUM *, BN_CTX *);
extern int BN_mod_sub(BIGNUM *, const BIGNUM *, const BIGNUM *, const BIGNUM *, BN_CTX *);
extern int BN_mod_inverse(BIGNUM *, const BIGNUM *, const BIGNUM *, BN_CTX *);
extern int BN_is_odd(const BIGNUM *);
}
#define cudaLimitStackSize 1
static inline cudaError_t cudaDeviceSetLimit(int l, size_t v) { return 0; }
