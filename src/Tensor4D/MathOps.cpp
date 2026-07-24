#include "Tensor4D\MathOps.hpp"
#include <cstddef>
#include <iostream>

void FORCE_INLINE MathOps::MatMul8x8Accumulate(float *dst, float *src0, float *src1)
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

float FORCE_INLINE HorizontalMax(__m256 value)
{
    // Compare the lower four floats with the upper four floats.
    __m128 low  = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 max4 = _mm_max_ps(low, high);

    // max4 = [max(0,4), max(1,5), max(2,6), max(3,7)]

    __m128 highPair = _mm_movehl_ps(max4, max4);
    max4 = _mm_max_ps(max4, highPair);

    // Compare lanes 0 and 1.
    __m128 lane1 = _mm_shuffle_ps(
        max4,
        max4,
        _MM_SHUFFLE(1, 1, 1, 1));

    max4 = _mm_max_ss(max4, lane1);

    return _mm_cvtss_f32(max4);
}

float FORCE_INLINE MathOps::HorizontalSum(__m256 value)
{
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);

    __m128 sum = _mm_add_ps(low, high);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

float MathOps::Dot(float *src0, float *src1, unsigned int size)
{
    unsigned int blockEnd = size - size % 64;
    unsigned int vectorEnd = size - size % 8;
    unsigned int k = 0;

    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    for (; k < blockEnd; k+=64)
    {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k), _mm256_loadu_ps(src1 + k), c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k + 8), _mm256_loadu_ps(src1 + k + 8), c1);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k + 16), _mm256_loadu_ps(src1 + k + 16), c2);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k + 24), _mm256_loadu_ps(src1 + k + 24), c3);
        c4 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k + 32), _mm256_loadu_ps(src1 + k + 32), c4);
        c5 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k + 40), _mm256_loadu_ps(src1 + k + 40), c5);
        c6 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k + 48), _mm256_loadu_ps(src1 + k + 48), c6);
        c7 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + k + 56), _mm256_loadu_ps(src1 + k + 56), c7);
    }

    for (; k < vectorEnd; k += 8)
    {
        c0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(src0 + k),
            _mm256_loadu_ps(src1 + k),
            c0);
    }

    c0 = _mm256_add_ps(c0, c1);
    c2 = _mm256_add_ps(c2, c3);
    c4 = _mm256_add_ps(c4, c5);
    c6 = _mm256_add_ps(c6, c7);

    c0 = _mm256_add_ps(c0, c2);
    c4 = _mm256_add_ps(c4, c6);

    c0 = _mm256_add_ps(c0, c4);

    
    float result = HorizontalSum(c0);

    for (; k < size; ++k)
    {
        result += src0[k] * src1[k];
    }

    return result;
}

void MathOps::MatVec(float *mat, float *vec, float *dst, unsigned int l, unsigned int c)
{
    for (unsigned int currentl = 0; currentl < l; currentl++)
    {
        dst[currentl] = Dot(mat + (currentl * c), vec, c);
    }
}

float MathOps::VecSum(float *src, unsigned int size)
{
    unsigned int alignedSize = (size / 8) * 8;
    float ret = 0;

    __m256 sRet = _mm256_setzero_ps();
    for (unsigned int i = 0; i < alignedSize; i+=8)
    {
        __m256 a = _mm256_loadu_ps(src + i);
        sRet  = _mm256_add_ps(sRet, a);
    }

    ret += HorizontalSum(sRet);

    for (unsigned int i = alignedSize; i < size; i++)
    {
        ret += src[i];
    }

    return ret;
}

void MathOps::ElementwiseMul(float *src, float val, float* dst, unsigned int size)
{
    unsigned int alignedSize = (size / 8) * 8;
    
    __m256 b = _mm256_set1_ps(val);
    for (unsigned int i = 0; i < alignedSize; i+=8)
    {
        __m256 a = _mm256_loadu_ps(src + i);
        __m256 result = _mm256_mul_ps(a, b);

        _mm256_storeu_ps(dst + i, result);
    }

    for (unsigned int i = alignedSize; i < size; i++)
    {
        dst[i] = src[i] * val;
    }
}

void MathOps::ElementwiseMul(float *src, float *src1, float* dst, unsigned int size)
{
    unsigned int alignedSize = (size / 8) * 8;

    for (unsigned int i = 0; i < alignedSize; i+=8)
    {
        __m256 a = _mm256_loadu_ps(src + i);
        __m256 b = _mm256_loadu_ps(src1 + i);
        __m256 result = _mm256_mul_ps(a, b);

        _mm256_storeu_ps(dst + i, result);
    }

    for (unsigned int i = alignedSize; i < size; i++)
    {
        dst[i] = src[i] * src1[i];
    }
}

void MathOps::VecAdd(float *src, float val, float *dst, unsigned int size)
{
    unsigned int alignedSize = (size / 8) * 8;
    
    __m256 b = _mm256_set1_ps(val);
    for (unsigned int i = 0; i < alignedSize; i+=8)
    {
        __m256 a = _mm256_loadu_ps(src + i);
        __m256 result = _mm256_add_ps(a, b);

        _mm256_storeu_ps(dst + i, result);
    }

    for (unsigned int i = alignedSize; i < size; i++)
    {
        dst[i] = src[i] + val;
    }
}

void MathOps::VecAdd(float *src, float *src1, float *dst, unsigned int size)
{
    unsigned int alignedSize = (size / 8) * 8;

    for (unsigned int i = 0; i < alignedSize; i+=8)
    {
        __m256 a = _mm256_loadu_ps(src + i);
        __m256 b = _mm256_loadu_ps(src1 + i);
        __m256 result = _mm256_add_ps(a, b);

        _mm256_storeu_ps(dst + i, result);
    }

    for (unsigned int i = alignedSize; i < size; i++)
    {
        dst[i] = src[i] + src1[i];
    }
}

void MathOps::Transpose(float *src, float *dst, unsigned int l, unsigned int c)
{
    for (unsigned int i = 0; i < l; i++)
    {
        for (unsigned int j = 0; j < c; j++)
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

void MathOps::CacheFriendly(float *src, float *dst, unsigned int l, unsigned int c)
{
    unsigned int dstOffset = 0;

    for (unsigned int currentL = 0; currentL < l; currentL+= 8)
    {
        for (unsigned int currentC = 0; currentC < c; currentC+= 8)
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

void MathOps::Softmax(float *src, float *dest, unsigned int size)
{
    float max = MathOps::Max(src, size);
    MathOps::VecAdd(src, -max, dest, size);

    for (unsigned int i = 0; i < size; i++)
    {
        dest[i] = std::exp(dest[i]);
    }

    //norm
    float sum = MathOps::VecSum(dest, size);
    MathOps::ElementwiseMul(dest, 1 / sum, dest, size);
}

float MathOps::Max(float* vec, unsigned int size)
{
    if (size == 0)
    {
        throw std::invalid_argument(
            "MathOps::Max requires at least one element.");
    }

    const unsigned int blockEnd  = size - size % 64;
    const unsigned int vectorEnd = size - size % 8;

    unsigned int k = 0;

    const __m256 negativeInfinity = _mm256_set1_ps(
        -std::numeric_limits<float>::infinity());

    __m256 c0 = negativeInfinity;
    __m256 c1 = negativeInfinity;
    __m256 c2 = negativeInfinity;
    __m256 c3 = negativeInfinity;
    __m256 c4 = negativeInfinity;
    __m256 c5 = negativeInfinity;
    __m256 c6 = negativeInfinity;
    __m256 c7 = negativeInfinity;

    // Process 64 floats per iteration.
    for (; k < blockEnd; k += 64)
    {
        c0 = _mm256_max_ps(c0, _mm256_loadu_ps(vec + k));
        c1 = _mm256_max_ps(c1, _mm256_loadu_ps(vec + k + 8));
        c2 = _mm256_max_ps(c2, _mm256_loadu_ps(vec + k + 16));
        c3 = _mm256_max_ps(c3, _mm256_loadu_ps(vec + k + 24));
        c4 = _mm256_max_ps(c4, _mm256_loadu_ps(vec + k + 32));
        c5 = _mm256_max_ps(c5, _mm256_loadu_ps(vec + k + 40));
        c6 = _mm256_max_ps(c6, _mm256_loadu_ps(vec + k + 48));
        c7 = _mm256_max_ps(c7, _mm256_loadu_ps(vec + k + 56));
    }

    // Process remaining complete vectors.
    for (; k < vectorEnd; k += 8)
    {
        c0 = _mm256_max_ps(
            c0,
            _mm256_loadu_ps(vec + k));
    }

    // Reduce eight vector accumulators into one.
    c0 = _mm256_max_ps(c0, c1);
    c2 = _mm256_max_ps(c2, c3);
    c4 = _mm256_max_ps(c4, c5);
    c6 = _mm256_max_ps(c6, c7);

    c0 = _mm256_max_ps(c0, c2);
    c4 = _mm256_max_ps(c4, c6);

    c0 = _mm256_max_ps(c0, c4);

    float result = HorizontalMax(c0);

    // Process the remaining 0–7 scalar values.
    for (; k < size; ++k)
    {
        if (vec[k] > result)
        {
            result = vec[k];
        }
    }

    return result;
}

void MathOps::ScaleAndReduce(float *mat, float *scaleVec, float *dest, unsigned int l, unsigned int c)
{

}

void MathOps::AccessFriendly(float *src, float *dst, unsigned int l, unsigned int c)
{
    unsigned int srcOffset = 0;

    for (unsigned int currentL = 0; currentL < l; currentL+= 8)
    {
        for (unsigned int currentC = 0; currentC < c; currentC+= 8)
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