#pragma once
#include "Tensor4D/Tensor4D.hpp"

namespace TensorUtils
{
    void RMSNorm(Tensor4D &tgt, Tensor4D &scales);

    void SiLU(Tensor4D &tgt);

    void GQA(Tensor4D& Q, Tensor4D& K, Tensor4D& V, Tensor4D& scoresTemp, Tensor4D& x, unsigned int position, unsigned int QHeadsPerKV, unsigned int KVHeads);
}