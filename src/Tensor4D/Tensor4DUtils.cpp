#include "Tensor4D/Tensor4DUtils.hpp"
#include "Tensor4D/MathOps.hpp"
#include <stdexcept>
#include <cmath>

void 

void TensorUtils::RMSNorm(Tensor4D& tgt, Tensor4D& scales)
{
    if(tgt.IsCacheFriendly())
        throw std::runtime_error("Invalid memory layout.");

    Shape s = tgt.GetShape();
    unsigned int channelSizeBytes = s.C * sizeof(float);

    if(s.C != scales.GetShape().C)
        throw std::runtime_error("Invalid shape");


    for(unsigned short int b0 = 0; b0 < s.B0; b0++)
    {
        for(unsigned short int b1 = 0; b1 < s.B1; b1++)
        {
            for(unsigned short int l = 0; l < s.L; l++)
            {
                float *temp = tgt.GetLCBuffer();
                float *vec = tgt.GetStorage(b0, b1, l, 0);

                //rms
                memcpy(temp, vec, channelSizeBytes);
                MathOps::ElementwiseMul(temp, temp, temp, s.C);
                float rms = std::sqrt(MathOps::VecSum(temp, s.C) / s.C);
                rms = 1 / rms;

                //val * (1/rms) * scale
                MathOps::ElementwiseMul(temp, rms, temp, s.C);
                MathOps::ElementwiseMul(temp, scales.GetStorage(), vec, s.C);
            }
        }
    }
}