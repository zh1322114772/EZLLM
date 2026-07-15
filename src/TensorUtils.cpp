#include "TensorUtils.hpp"
#include <math.h>

float TensorUtils::ChannelDot(Tensor& m0, Tensor& m1, unsigned int b0, unsigned int l0, unsigned int b1, unsigned int l1)
{
    unsigned int m0C = m0.GetShape().channels;

    if(m0C != m1.GetShape().channels)
    {
        throw std::runtime_error("Invalid shape.");
    }


    float ret = 0;

    for (unsigned int i = 0; i < m0C; i++)
    {
        ret += m0.GetValue(b0, l0, i) * m1.GetValue(b1, l1, i);
    }

    return ret;
}

void TensorUtils::ComputeRoPEPositions(PositionMemory &mem, unsigned int position)
{
    float theta = 100000;

    for (unsigned int i = 0; i < 32; i++)
    {
        float inv_freq = 1 / (std::pow(theta, (float)(2 * i) / 64));
        float angle = (float)position * inv_freq;

        mem.sin.SetValue(0, 0, i, std::sin(angle));
        mem.cos.SetValue(0, 0, i, std::cos(angle));
    }
}