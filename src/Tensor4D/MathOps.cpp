#include "Tensor4D\MathOps.hpp"
#include <immintrin.h>
#include <cstddef>
#include <iostream>

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

void MathOps::Transpose(float *src, float *dst, unsigned short int l, unsigned short int c)
{
    for (unsigned short int i = 0; i < l; i++)
    {
        for (unsigned short int j = 0; j < c; j++)
        {
            dst[(j * l) + c] = src[(i * c) + j];
        }
    }
}

void MathOps::MatMul88(float *m0, float *m1, float *dest, unsigned int s0, unsigned int s1, unsigned int s2)
{
    unsigned int tgtL = s0 / 8;
    unsigned int tgtI = s1 / 8;
    unsigned int tgtC = s2 / 8;

    for (unsigned int l = 0; l < tgtL; l++)
    {
        for (unsigned int i = 0; i < tgtI; i++)
        {
            float *m0Offset = m0 + (l * tgtI * 64) + (i * 64);

            for (unsigned int c = 0; c < tgtC; c++)
            {
                float *m1Offset = m1 + (i * tgtC * 64) + (c * 64);
                float *tgtOffset = dest + (l * tgtC * 64) + (c * 64);

                MatMul8x8Accumulate(tgtOffset, m0Offset, m1Offset);
            }
        }
    }
}

void MathOps::CacheFriendly(float *src, float *dst, unsigned short int l, unsigned short int c)
{
    unsigned int dstOffset = 0;

    for (unsigned short int currentL = 0; currentL < l; currentL+= 8)
    {
        for (unsigned short int currentC = 0; currentC < c; currentC+= 8)
        {
            __m256 c0 = _mm256_loadu_ps(src + ((currentL) * c) + currentC);
            __m256 c1 = _mm256_loadu_ps(src + ((currentL + 1) * c) + currentC);
            __m256 c2 = _mm256_loadu_ps(src + ((currentL + 2) * c) + currentC);
            __m256 c3 = _mm256_loadu_ps(src + ((currentL + 3) * c) + currentC);
            __m256 c4 = _mm256_loadu_ps(src + ((currentL + 4) * c) + currentC);
            __m256 c5 = _mm256_loadu_ps(src + ((currentL + 5) * c) + currentC);
            __m256 c6 = _mm256_loadu_ps(src + ((currentL + 6) * c) + currentC);
            __m256 c7 = _mm256_loadu_ps(src + ((currentL + 7) * c) + currentC);
            
            _mm256_storeu_ps(dst + dstOffset,  c0);
            _mm256_storeu_ps(dst + dstOffset + 8,  c1);
            _mm256_storeu_ps(dst + dstOffset + 16, c2);
            _mm256_storeu_ps(dst + dstOffset + 24, c3);
            _mm256_storeu_ps(dst + dstOffset + 32, c4);
            _mm256_storeu_ps(dst + dstOffset + 40, c5);
            _mm256_storeu_ps(dst + dstOffset + 48, c6);
            _mm256_storeu_ps(dst + dstOffset + 56, c7);

            dstOffset += 64;
        }
    }
}


void MathOps::AccessFriendly(float *src, float *dst, unsigned short int l, unsigned short int c)
{
    unsigned int srcOffset = 0;

    for (unsigned short int currentL = 0; currentL < l; currentL+= 8)
    {
        for (unsigned short int currentC = 0; currentC < c; currentC+= 8)
        {
            __m256 c0 = _mm256_loadu_ps(src + srcOffset);
            __m256 c1 = _mm256_loadu_ps(src + srcOffset + 8);
            __m256 c2 = _mm256_loadu_ps(src + srcOffset + 16);
            __m256 c3 = _mm256_loadu_ps(src + srcOffset + 24);
            __m256 c4 = _mm256_loadu_ps(src + srcOffset + 32);
            __m256 c5 = _mm256_loadu_ps(src + srcOffset + 40);
            __m256 c6 = _mm256_loadu_ps(src + srcOffset + 48);
            __m256 c7 = _mm256_loadu_ps(src + srcOffset + 56);
            
            _mm256_storeu_ps(dst + ((currentL) * c) + currentC, c0);
            _mm256_storeu_ps(dst + ((currentL + 1) * c) + currentC, c1);
            _mm256_storeu_ps(dst + ((currentL + 2) * c) + currentC, c2);
            _mm256_storeu_ps(dst + ((currentL + 3) * c) + currentC, c3);
            _mm256_storeu_ps(dst + ((currentL + 4) * c) + currentC, c4);
            _mm256_storeu_ps(dst + ((currentL + 5) * c) + currentC, c5);
            _mm256_storeu_ps(dst + ((currentL + 6) * c) + currentC, c6);
            _mm256_storeu_ps(dst + ((currentL + 7) * c) + currentC, c7);

            srcOffset += 64;
        }
    }
}