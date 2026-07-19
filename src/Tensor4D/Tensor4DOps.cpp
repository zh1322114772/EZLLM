#include "Tensor4D\Tensor4D.hpp"
#include <immintrin.h>
#include <cstddef>

#if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE inline __attribute__((always_inline))
#else
    #define FORCE_INLINE inline
#endif

void FORCE_INLINE MatMul8x8Accumulate(float *dst, float *src0, float *src1)
{
    __m256 c0 = _mm256_loadu_ps(dst + 0);
    __m256 c1 = _mm256_loadu_ps(dst + 8);
    __m256 c2 = _mm256_loadu_ps(dst + 16);
    __m256 c3 = _mm256_loadu_ps(dst + 24);
    __m256 c4 = _mm256_loadu_ps(dst + 32);
    __m256 c5 = _mm256_loadu_ps(dst + 40);
    __m256 c6 = _mm256_loadu_ps(dst + 48);
    __m256 c7 = _mm256_loadu_ps(dst + 56);

    for (int k = 0; k < 8; ++k)
    {
        const __m256 bk = _mm256_loadu_ps(src1 + k * 8);

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(src0[0 * 8 + k]), bk, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(src0[1 * 8 + k]), bk, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(src0[2 * 8 + k]), bk, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(src0[3 * 8 + k]), bk, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(src0[4 * 8 + k]), bk, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(src0[5 * 8 + k]), bk, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(src0[6 * 8 + k]), bk, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(src0[7 * 8 + k]), bk, c7);
    }

    _mm256_storeu_ps(dst + 0,  c0);
    _mm256_storeu_ps(dst + 8,  c1);
    _mm256_storeu_ps(dst + 16, c2);
    _mm256_storeu_ps(dst + 24, c3);
    _mm256_storeu_ps(dst + 32, c4);
    _mm256_storeu_ps(dst + 40, c5);
    _mm256_storeu_ps(dst + 48, c6);
    _mm256_storeu_ps(dst + 56, c7);
}

void Tensor4D::Transpose(float *src, float *dst, unsigned short int l, unsigned short int c)
{

}

void Tensor4D::MatMul88(float *m0, float *m1, float *dest, unsigned int s0, unsigned int s1, unsigned int s2)
{
    for (unsigned int currentS0 = 0; currentS0 < s0; currentS0 += 8)
    {
        for (unsigned int currentS1 = 0; currentS1 < s1; currentS1 += 8)
        {
            for (unsigned int currentS2 = 0; currentS2 < s2; currentS2 += 8)
            {
                float *m0Offset = m0 + (currentS0 * s1 * 64) + (currentS1 * 64);
                float *m1Offset = m1 + (currentS1 * s2 * 64) + (currentS2 * 64);
                float *tgtOffset = dest + (currentS0 * s1 * 64) + (currentS2 * 64);

                MatMul8x8Accumulate(tgtOffset, m0Offset, m1Offset);
            }
        }
    }
}

void Tensor4D::CacheFriendly(float *src, float *dst, unsigned short int l, unsigned short int c)
{

}

void Tensor4D::AccessFriendly(float *src, float *dst, unsigned short int l, unsigned short int c)
{

}