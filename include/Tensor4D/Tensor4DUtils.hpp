#pragma once
#include "Tensor4D/Tensor4D.hpp"

namespace TensorUtils
{
    void RMSNorm(Tensor4D &tgt, Tensor4D &scales);
}