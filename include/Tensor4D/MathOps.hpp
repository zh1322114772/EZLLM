#pragma once

#if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE inline __attribute__((always_inline))
#else
    #define FORCE_INLINE inline
#endif

namespace MathOps
{
    void Transpose(float *src, float *dst, unsigned short int l, unsigned short int c);

    void MatMul88(float *m0, float *m1, float *dest, unsigned int s0,  unsigned int s1,  unsigned int s2);

    //void BatchMul88(Tensor4D &m0, Tensor4D &m1, unsigned int offsetm0, unsigned int offsetm1);
    //to be implemeneted

    void VecAdd(float *src0, float *src1, float *dst, unsigned int size);

    void VecAdd(float *src0, float val, float *dst, unsigned int size);

    float VecSum(float *src, unsigned int size);

    void ElementwiseMul(float *src, float val, float* dst, unsigned int size);

    void ElementwiseMul(float *src, float *src1, float* dst, unsigned int size);

    void CacheFriendly(float *src, float *dst, unsigned short int l, unsigned short int c);

    void AccessFriendly(float *src, float *dst, unsigned short int l, unsigned short int c);
}
