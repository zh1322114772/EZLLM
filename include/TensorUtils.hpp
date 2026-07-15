#pragma once

#include "Tensor.hpp"
#include "PositionMemory.hpp"

namespace TensorUtils
{
    float ChannelDot(Tensor& m0, Tensor& m1, unsigned int b0, unsigned int l0, unsigned int b1, unsigned int l1);

    void ComputeRoPEPositions(PositionMemory &mem, unsigned int position);
}