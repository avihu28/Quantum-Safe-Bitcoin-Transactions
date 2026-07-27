/*
* This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
* Copyright (c) 2019 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------


#define GRP_SIZE (1024*2)

#define HSIZE ((GRP_SIZE / 2) - 1)

// 64bits lsb negative inverse of P (mod 2^64)
#define MM64 0xD838091DD2253531ULL


// We need 1 extra block for ModInv
#define NBBLOCK 5
#define BIFULLSIZE 40

// Assembly directives
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDS(r,a,b,c) asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

// SECPK1 endomorphism constants
//__device__ __constant__ uint64_t _beta[] = { 0xC1396C28719501EEULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL };
//__device__ __constant__ uint64_t _beta2[] = { 0x3EC693D68E6AFA40ULL, 0x630FB68AED0A766AULL, 0x919BB86153CBCB16ULL, 0x851695D49A83F8EFULL };


// ---------------------------------------------------------------------------------------

#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define _IsEqual(a,b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)     ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)      ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

#define __sright128(a,b,n) ((a)>>(n))|((b)<<(64-(n)))
#define __sleft128(a,b,n) ((b)<<(n))|((a)>>(64-(n)))

// ---------------------------------------------------------------------------------------

#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Add2(r,a,b)  {\
  UADDO(r[0], a[0], b[0]); \
  UADDC(r[1], a[1], b[1]); \
  UADDC(r[2], a[2], b[2]); \
  UADDC(r[3], a[3], b[3]); \
  UADD(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Sub2(r,a,b)  {\
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define Sub1(r,a) {\
  USUBO1(r[0], a[0]); \
  USUBC1(r[1], a[1]); \
  USUBC1(r[2], a[2]); \
  USUBC1(r[3], a[3]); \
  USUB1(r[4], a[4]);}

// ---------------------------------------------------------------------------------------

#define Neg(r) {\
USUBO(r[0],0ULL,r[0]); \
USUBC(r[1],0ULL,r[1]); \
USUBC(r[2],0ULL,r[2]); \
USUBC(r[3],0ULL,r[3]); \
USUB(r[4],0ULL,r[4]); }

// ---------------------------------------------------------------------------------------

#define UMult(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL);}

// ---------------------------------------------------------------------------------------

#define Load(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; \
  (r)[4] = (a)[4];}

// ---------------------------------------------------------------------------------------

#define _LoadI64(r, a) {\
  (r)[0] = a; \
  (r)[1] = a>>63; \
  (r)[2] = (r)[1]; \
  (r)[3] = (r)[1]; \
  (r)[4] = (r)[1];}
// ---------------------------------------------------------------------------------------

#define Load256(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3];}

// ---------------------------------------------------------------------------------------

#define Load256A(r, a) {\
  (r)[0] = (a)[IDX]; \
  (r)[1] = (a)[IDX+blockDim.x]; \
  (r)[2] = (a)[IDX+2*blockDim.x]; \
  (r)[3] = (a)[IDX+3*blockDim.x];}

// ---------------------------------------------------------------------------------------

#define Store256A(r, a) {\
  (r)[IDX] = (a)[0]; \
  (r)[IDX+blockDim.x] = (a)[1]; \
  (r)[IDX+2*blockDim.x] = (a)[2]; \
  (r)[IDX+3*blockDim.x] = (a)[3];}

// ---------------------------------------------------------------------------------------

__device__ void _ShiftR62(uint64_t *r)
{

    r[0] = (r[1] << 2) | (r[0] >> 62);
    r[1] = (r[2] << 2) | (r[1] >> 62);
    r[2] = (r[3] << 2) | (r[2] >> 62);
    r[3] = (r[4] << 2) | (r[3] >> 62);
    // With sign extent
    r[4] = (int64_t)(r[4]) >> 62;

}

__device__ void _ShiftR62(uint64_t dest[5], uint64_t r[5], uint64_t carry)
{

    dest[0] = (r[1] << 2) | (r[0] >> 62);
    dest[1] = (r[2] << 2) | (r[1] >> 62);
    dest[2] = (r[3] << 2) | (r[2] >> 62);
    dest[3] = (r[4] << 2) | (r[3] >> 62);
    dest[4] = (carry << 2) | (r[4] >> 62);

}

// ---------------------------------------------------------------------------------------

__device__ void _IMult(uint64_t *r, uint64_t *a, int64_t b)
{

    uint64_t t[NBBLOCK];

    // Make b positive
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    } else {
        Load(t, a);
    }

    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADD(r[4], t[3], b, r[4]);

}

__device__ uint64_t _IMultC(uint64_t *r, uint64_t *a, int64_t b)
{

    uint64_t t[NBBLOCK];
    uint64_t carry;

    // Make b positive
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    } else {
        Load(t, a);
    }

    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADDC(r[4], t[3], b, r[4]);
    MADDS(carry, t[4], b, 0ULL);

    return carry;

}

// ---------------------------------------------------------------------------------------

__device__ void _MulP(uint64_t *r, uint64_t a)
{

    uint64_t ah;
    uint64_t al;

    UMULLO(al, a, 0x1000003D1ULL);
    UMULHI(ah, a, 0x1000003D1ULL);

    USUBO(r[0], 0ULL, al);
    USUBC(r[1], 0ULL, ah);
    USUBC(r[2], 0ULL, 0ULL);
    USUBC(r[3], 0ULL, 0ULL);
    USUB(r[4], a, 0ULL);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModNeg256(uint64_t *r, uint64_t *a)
{

    uint64_t t[4];
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModNeg256(uint64_t *r)
{

    uint64_t t[4];
    USUBO(t[0], 0ULL, r[0]);
    USUBC(t[1], 0ULL, r[1]);
    USUBC(t[2], 0ULL, r[2]);
    USUBC(t[3], 0ULL, r[3]);
    UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);

}

__device__ void _ModAdd256(uint64_t *r, uint64_t *a, uint64_t *b)
{
    uint64_t rr[5];

    UADDO(rr[0], a[0], b[0]);
    UADDC(rr[1], a[1], b[1]);
    UADDC(rr[2], a[2], b[2]);
    UADDC(rr[3], a[3], b[3]);
    UADD(rr[4], 0UL, 0UL);

    Load256(r, rr);

    SubP(rr);

    if(_IsPositive(rr)) {
        Load256(r, rr);
    }
}

__device__ void _ModSub256(uint64_t *r, uint64_t *a, uint64_t *b)
{
    uint64_t t;
    uint64_t T[4];
    
    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(t, 0ULL, 0ULL);

    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;

    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModSub256(uint64_t *r, uint64_t *b)
{

    uint64_t t;
    uint64_t T[4];
    USUBO(r[0], r[0], b[0]);
    USUBC(r[1], r[1], b[1]);
    USUBC(r[2], r[2], b[2]);
    USUBC(r[3], r[3], b[3]);
    USUB(t, 0ULL, 0ULL);
    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t _CTZ(uint64_t x)
{
    uint32_t n;
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " brev.b64 tmp, %1;\n\t"
        " clz.b64 %0, tmp;\n\t"
        "}"
        : "=r"(n) : "l"(x));
    return n;
}

// ---------------------------------------------------------------------------------------
#define SWAP(tmp,x,y) tmp = x; x = y; y = tmp;
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ void _DivStep62(uint64_t u[5], uint64_t v[5],
                           int32_t *pos,
                           int64_t *uu, int64_t *uv,
                           int64_t *vu, int64_t *vv)
{


    // u' = (uu*u + uv*v) >> bitCount
    // v' = (vu*u + vv*v) >> bitCount
    // Do not maintain a matrix for r and s, the number of
    // 'added P' can be easily calculated

    *uu = 1; *uv = 0;
    *vu = 0; *vv = 1;

    uint32_t bitCount = 62;
    uint32_t zeros;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];

    // Extract 64 MSB of u and v
    // u and v must be positive
    uint64_t uh, vh;
    int64_t w, x, y, z;
    bitCount = 62;

    while (*pos > 0 && (u[*pos] | v[*pos]) == 0)
        (*pos)--;
    if (*pos == 0) {

        uh = u[0];
        vh = v[0];

    } else {

        uint32_t s = __clzll(u[*pos] | v[*pos]);
        if (s == 0) {
            uh = u[*pos];
            vh = v[*pos];
        } else {
            uh = __sleft128(u[*pos - 1], u[*pos], s);
            vh = __sleft128(v[*pos - 1], v[*pos], s);
        }

    }


    while (true) {

        // Use a sentinel bit to count zeros only up to bitCount
        zeros = _CTZ(v0 | (1ULL << bitCount));

        v0 >>= zeros;
        vh >>= zeros;
        *uu <<= zeros;
        *uv <<= zeros;
        bitCount -= zeros;

        if (bitCount == 0)
            break;

        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(x, u0, v0);
            SWAP(y, *uu, *vu);
            SWAP(z, *uv, *vv);
        }

        vh -= uh;
        v0 -= u0;
        *vv -= *uv;
        *vu -= *uu;

    }

}

__device__ void _MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, uint64_t *carry)
{

    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t c1, c2;

    c1 = _IMultC(t1, u, _11);
    c2 = _IMultC(t2, v, _12);

    UADDO(dest[0], t1[0], t2[0]);
    UADDC(dest[1], t1[1], t2[1]);
    UADDC(dest[2], t1[2], t2[2]);
    UADDC(dest[3], t1[3], t2[3]);
    UADDC(dest[4], t1[4], t2[4]);
    UADD(*carry, c1, c2);

}

__device__ void _MatrixVecMul(uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, int64_t _21, int64_t _22)
{

    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t t3[NBBLOCK];
    uint64_t t4[NBBLOCK];

    _IMult(t1, u, _11);
    _IMult(t2, v, _12);
    _IMult(t3, u, _21);
    _IMult(t4, v, _22);

    UADDO(u[0], t1[0], t2[0]);
    UADDC(u[1], t1[1], t2[1]);
    UADDC(u[2], t1[2], t2[2]);
    UADDC(u[3], t1[3], t2[3]);
    UADD(u[4], t1[4], t2[4]);

    UADDO(v[0], t3[0], t4[0]);
    UADDC(v[1], t3[1], t4[1]);
    UADDC(v[2], t3[2], t4[2]);
    UADDC(v[3], t3[3], t4[3]);
    UADD(v[4], t3[4], t4[4]);

}

__device__ uint64_t _AddCh(uint64_t r[5], uint64_t a[5], uint64_t carry)
{

    uint64_t carryOut;

    UADDO1(r[0], a[0]);
    UADDC1(r[1], a[1]);
    UADDC1(r[2], a[2]);
    UADDC1(r[3], a[3]);
    UADDC1(r[4], a[4]);
    UADD(carryOut, carry, 0ULL);

    return carryOut;

}

__device__ __noinline__ void _ModInv(uint64_t *R)
{

    // Compute modular inverse of R mod P (using 320bits signed integer)
    // 0 < this < P  , P must be odd
    // Return 0 if no inverse
    // See IntMod.cpp for more info.

    uint64_t u[NBBLOCK];
    uint64_t v[NBBLOCK];
    uint64_t r[NBBLOCK];
    uint64_t s[NBBLOCK];
    uint64_t tr[NBBLOCK];
    uint64_t ts[NBBLOCK];
    uint64_t r0[NBBLOCK];
    uint64_t s0[NBBLOCK];

    int64_t  uu;
    int64_t  uv;
    int64_t  vu;
    int64_t  vv;

    uint64_t mr0;
    uint64_t ms0;

    uint64_t carryR;
    uint64_t carryS;

    int32_t  pos = NBBLOCK - 1;

    u[0] = 0xFFFFFFFEFFFFFC2F;
    u[1] = 0xFFFFFFFFFFFFFFFF;
    u[2] = 0xFFFFFFFFFFFFFFFF;
    u[3] = 0xFFFFFFFFFFFFFFFF;
    u[4] = 0;
    Load(v, R);
    r[0] = 0; s[0] = 1;
    r[1] = 0; s[1] = 0;
    r[2] = 0; s[2] = 0;
    r[3] = 0; s[3] = 0;
    r[4] = 0; s[4] = 0;

    // Delayed right shift 62bits

    // DivStep loop -------------------------------

    while (true) {

        _DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);

        _MatrixVecMul(u, v, uu, uv, vu, vv);

        if (_IsNegative(u)) {
            Neg(u);
            uu = -uu;
            uv = -uv;
        }
        if (_IsNegative(v)) {
            Neg(v);
            vu = -vu;
            vv = -vv;
        }

        _ShiftR62(u);
        _ShiftR62(v);

        // Update r
        _MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
        mr0 = (tr[0] * MM64) & MSK62;
        _MulP(r0, mr0);
        carryR = _AddCh(tr, r0, carryR);

        if (_IsZero(v)) {

            _ShiftR62(r, tr, carryR);
            break;

        } else {

            // Update s
            _MatrixVecMulHalf(ts, r, s, vu, vv, &carryS);
            ms0 = (ts[0] * MM64) & MSK62;
            _MulP(s0, ms0);
            carryS = _AddCh(ts, s0, carryS);

        }

        _ShiftR62(r, tr, carryR);
        _ShiftR62(s, ts, carryS);

    }

    // u ends with gcd
    if (!_IsOne(u)) {
        // No inverse
        R[0] = 0ULL;
        R[1] = 0ULL;
        R[2] = 0ULL;
        R[3] = 0ULL;
        R[4] = 0ULL;
        return;
    }

    while (_IsNegative(r))
        AddP(r);
    while (!_IsNegative(r))
        SubP(r);
    AddP(r);

    Load(R, r);

}

// ---------------------------------------------------------------------------------------
// Compute a*b*(mod n)
// a and b must be lower than n
// ---------------------------------------------------------------------------------------
__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b)
{

    uint64_t r512[8];
    uint64_t t[NBBLOCK];
    uint64_t ah, al;

    r512[5] = 0;
    r512[6] = 0;
    r512[7] = 0;

    // 256*256 multiplier
    UMult(r512, a, b[0]);
    UMult(t, a, b[1]);
    UADDO1(r512[1], t[0]);
    UADDC1(r512[2], t[1]);
    UADDC1(r512[3], t[2]);
    UADDC1(r512[4], t[3]);
    UADD1(r512[5], t[4]);
    UMult(t, a, b[2]);
    UADDO1(r512[2], t[0]);
    UADDC1(r512[3], t[1]);
    UADDC1(r512[4], t[2]);
    UADDC1(r512[5], t[3]);
    UADD1(r512[6], t[4]);
    UMult(t, a, b[3]);
    UADDO1(r512[3], t[0]);
    UADDC1(r512[4], t[1]);
    UADDC1(r512[5], t[2]);
    UADDC1(r512[6], t[3]);
    UADD1(r512[7], t[4]);

    // Reduce from 512 to 320
    UMult(t, (r512 + 4), 0x1000003D1ULL);
    UADDO1(r512[0], t[0]);
    UADDC1(r512[1], t[1]);
    UADDC1(r512[2], t[2]);
    UADDC1(r512[3], t[3]);

    // Reduce from 320 to 256
    UADD1(t[4], 0ULL);
    UMULLO(al, t[4], 0x1000003D1ULL);
    UMULHI(ah, t[4], 0x1000003D1ULL);
    UADDO(r[0], r512[0], al);
    UADDC(r[1], r512[1], ah);
    UADDC(r[2], r512[2], 0ULL);
    UADD(r[3], r512[3], 0ULL);

}

__device__ void _ModMult(uint64_t *r, uint64_t *a)
{

    uint64_t r512[8];
    uint64_t t[NBBLOCK];
    uint64_t ah, al;
    r512[5] = 0;
    r512[6] = 0;
    r512[7] = 0;

    // 256*256 multiplier
    UMult(r512, a, r[0]);
    UMult(t, a, r[1]);
    UADDO1(r512[1], t[0]);
    UADDC1(r512[2], t[1]);
    UADDC1(r512[3], t[2]);
    UADDC1(r512[4], t[3]);
    UADD1(r512[5], t[4]);
    UMult(t, a, r[2]);
    UADDO1(r512[2], t[0]);
    UADDC1(r512[3], t[1]);
    UADDC1(r512[4], t[2]);
    UADDC1(r512[5], t[3]);
    UADD1(r512[6], t[4]);
    UMult(t, a, r[3]);
    UADDO1(r512[3], t[0]);
    UADDC1(r512[4], t[1]);
    UADDC1(r512[5], t[2]);
    UADDC1(r512[6], t[3]);
    UADD1(r512[7], t[4]);

    // Reduce from 512 to 320
    UMult(t, (r512 + 4), 0x1000003D1ULL);
    UADDO1(r512[0], t[0]);
    UADDC1(r512[1], t[1]);
    UADDC1(r512[2], t[2]);
    UADDC1(r512[3], t[3]);

    // Reduce from 320 to 256
    UADD1(t[4], 0ULL);
    UMULLO(al, t[4], 0x1000003D1ULL);
    UMULHI(ah, t[4], 0x1000003D1ULL);
    UADDO(r[0], r512[0], al);
    UADDC(r[1], r512[1], ah);
    UADDC(r[2], r512[2], 0ULL);
    UADD(r[3], r512[3], 0ULL);

}

__device__ void _ModSqr(uint64_t *rp, const uint64_t *up)
{

    uint64_t r512[8];

    uint64_t u10, u11;

    uint64_t r0;
    uint64_t r1;
    uint64_t r3;
    uint64_t r4;

    uint64_t t1;
    uint64_t t2;


    //k=0
    UMULLO(r512[0], up[0], up[0]);
    UMULHI(r1, up[0], up[0]);

    //k=1
    UMULLO(r3, up[0], up[1]);
    UMULHI(r4, up[0], up[1]);
    UADDO1(r3, r3);
    UADDC1(r4, r4);
    UADD(t1, 0x0ULL, 0x0ULL);
    UADDO1(r3, r1);
    UADDC1(r4, 0x0ULL);
    UADD1(t1, 0x0ULL);
    r512[1] = r3;

    //k=2
    UMULLO(r0, up[0], up[2]);
    UMULHI(r1, up[0], up[2]);
    UADDO1(r0, r0);
    UADDC1(r1, r1);
    UADD(t2, 0x0ULL, 0x0ULL);
    UMULLO(u10, up[1], up[1]);
    UMULHI(u11, up[1], up[1]);
    UADDO1(r0, u10);
    UADDC1(r1, u11);
    UADD1(t2, 0x0ULL);
    UADDO1(r0, r4);
    UADDC1(r1, t1);
    UADD1(t2, 0x0ULL);

    r512[2] = r0;

    //k=3
    UMULLO(r3, up[0], up[3]);
    UMULHI(r4, up[0], up[3]);
    UMULLO(u10, up[1], up[2]);
    UMULHI(u11, up[1], up[2]);
    UADDO1(r3, u10);
    UADDC1(r4, u11);
    UADD(t1, 0x0ULL, 0x0ULL);
    t1 += t1;
    UADDO1(r3, r3);
    UADDC1(r4, r4);
    UADD1(t1, 0x0ULL);
    UADDO1(r3, r1);
    UADDC1(r4, t2);
    UADD1(t1, 0x0ULL);

    r512[3] = r3;

    //k=4
    UMULLO(r0, up[1], up[3]);
    UMULHI(r1, up[1], up[3]);
    UADDO1(r0, r0);
    UADDC1(r1, r1);
    UADD(t2, 0x0ULL, 0x0ULL);
    UMULLO(u10, up[2], up[2]);
    UMULHI(u11, up[2], up[2]);
    UADDO1(r0, u10);
    UADDC1(r1, u11);
    UADD1(t2, 0x0ULL);
    UADDO1(r0, r4);
    UADDC1(r1, t1);
    UADD1(t2, 0x0ULL);

    r512[4] = r0;

    //k=5
    UMULLO(r3, up[2], up[3]);
    UMULHI(r4, up[2], up[3]);
    UADDO1(r3, r3);
    UADDC1(r4, r4);
    UADD(t1, 0x0ULL, 0x0ULL);
    UADDO1(r3, r1);
    UADDC1(r4, t2);
    UADD1(t1, 0x0ULL);

    r512[5] = r3;

    //k=6
    UMULLO(r0, up[3], up[3]);
    UMULHI(r1, up[3], up[3]);
    UADDO1(r0, r4);
    UADD1(r1, t1);
    r512[6] = r0;

    //k=7
    r512[7] = r1;

#if 1

    // Reduce from 512 to 320
    UMULLO(r0, r512[4], 0x1000003D1ULL);
    UMULLO(r1, r512[5], 0x1000003D1ULL);
    MADDO(r1, r512[4], 0x1000003D1ULL, r1);
    UMULLO(t2, r512[6], 0x1000003D1ULL);
    MADDC(t2, r512[5], 0x1000003D1ULL, t2);
    UMULLO(r3, r512[7], 0x1000003D1ULL);
    MADDC(r3, r512[6], 0x1000003D1ULL, r3);
    MADD(r4, r512[7], 0x1000003D1ULL, 0ULL);

    UADDO1(r512[0], r0);
    UADDC1(r512[1], r1);
    UADDC1(r512[2], t2);
    UADDC1(r512[3], r3);

    // Reduce from 320 to 256
    UADD1(r4, 0ULL);
    UMULLO(u10, r4, 0x1000003D1ULL);
    UMULHI(u11, r4, 0x1000003D1ULL);
    UADDO(rp[0], r512[0], u10);
    UADDC(rp[1], r512[1], u11);
    UADDC(rp[2], r512[2], 0ULL);
    UADD(rp[3], r512[3], 0ULL);

#else

    uint64_t z1, z2, z3, z4, z5, z6, z7, z8;

    UMULLO(z3, r512[5], 0x1000003d1ULL);
    UMULHI(z4, r512[5], 0x1000003d1ULL);
    UMULLO(z5, r512[6], 0x1000003d1ULL);
    UMULHI(z6, r512[6], 0x1000003d1ULL);
    UMULLO(z7, r512[7], 0x1000003d1ULL);
    UMULHI(z8, r512[7], 0x1000003d1ULL);
    UMULLO(z1, r512[4], 0x1000003d1ULL);
    UMULHI(z2, r512[4], 0x1000003d1ULL);
    UADDO1(z1, r512[0]);
    UADD1(z2, 0x0ULL);


    UADDO1(z2, r512[1]);
    UADDC1(z4, r512[2]);
    UADDC1(z6, r512[3]);
    UADD1(z8, 0x0ULL);

    UADDO1(z3, z2);
    UADDC1(z5, z4);
    UADDC1(z7, z6);
    UADD1(z8, 0x0ULL);

    UMULLO(u10, z8, 0x1000003d1ULL);
    UMULHI(u11, z8, 0x1000003d1ULL);
    UADDO1(z1, u10);
    UADDC1(z3, u11);
    UADDC1(z5, 0x0ULL);
    UADD1(z7, 0x0ULL);

    rp[0] = z1;
    rp[1] = z3;
    rp[2] = z5;
    rp[3] = z7;

#endif

}

//Very efficient way of finding 8-byte target value in global memory buffer (Buffer must be ordered in ascending order)
//Each step it does fast division by half: mid = (hi + lo) >> 1; and checks resulting value
//Worst-case performance is O(log n), and we don't need to calculate any hashes by using this method.
__device__ int _BinarySearch(uint64_t *buffer, int hi, uint64_t target)
{
    int mid;
	int lo = 0;

	while (hi - lo > 1)
	{
		mid = (hi + lo) >> 1;
		if (buffer[mid] == target)
		{
			return mid;
		}
		else if (buffer[mid] < target)
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid;
		}
	}
  
	if (buffer[lo] == target)
	{
		return lo;
	}
	else if (buffer[hi] == target)
	{
		return hi;
	}
	else
	{
		return -1;
	}
}

//Secp256k1 Point Addition implementation
__device__ void _PointAddSecp256k1(uint64_t *p1x, uint64_t *p1y, uint64_t *p1z, uint64_t *p2x, uint64_t *p2y)
{
  uint64_t u[4];
  uint64_t v[4];

  uint64_t us2[4];
  uint64_t vs2[4];
  uint64_t vs3[4];

  uint64_t a[4];

  uint64_t us2w[4];
  uint64_t vs2v2[4];
  uint64_t vs3u2[4];
  uint64_t _2vs2v2[4];

  _ModMult(u, p2y, p1z);
  _ModMult(v, p2x, p1z);

  _ModSub256(u, u, p1y);
  _ModSub256(v, v, p1x);

  _ModSqr(us2, u);
  _ModSqr(vs2, v);

  _ModMult(vs3, vs2, v);
  _ModMult(us2w, us2, p1z);
  _ModMult(vs2v2, vs2, p1x);

  _ModAdd256(_2vs2v2, vs2v2, vs2v2);

  _ModSub256(a, us2w, vs3);
  _ModSub256(a, _2vs2v2);

  _ModMult(p1x, v, a);
  _ModMult(vs3u2, vs3, p1y);

  _ModSub256(p1y, vs2v2, a);
  _ModMult(p1y, p1y, u);

  _ModSub256(p1y, vs3u2);
  _ModMult(p1z, vs3, p1z);
}



