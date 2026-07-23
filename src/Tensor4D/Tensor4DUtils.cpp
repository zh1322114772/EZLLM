#include "Tensor4D/Tensor4DUtils.hpp"
#include "Tensor4D/MathOps.hpp"
#include <stdexcept>
#include <cmath>

/**
 * void LayerBlock::GroupedQueryAttention(LayerMemory &memory, Tensor &q, Tensor &x, unsigned int position)
{
    x.Reshape(1, 15, 64);

    unsigned int steps = std::min(position + 1, CONTEXT_WINDOW);
    unsigned int beginOffset = ((position + 1) - steps) % CONTEXT_WINDOW;

    unsigned int nq = 15;
    unsigned int nkv = 5;
    unsigned int numQPerKv = nq / nkv;

    for (unsigned int qhead = 0; qhead < nq; qhead++)
    {
        unsigned kvhead = qhead / numQPerKv;


        //compute scores
        float maxScore = std::numeric_limits<float>::lowest();
        float totalScore = 0;

        for (unsigned int currentStep = 0; currentStep < steps; currentStep++)
        {
            unsigned int adjustedBatch = (currentStep + beginOffset) % CONTEXT_WINDOW;
            float currentScore = TensorUtils::ChannelDot(q, memory.KCache, 0, qhead, adjustedBatch, kvhead) / 8;
            memory.Scores[adjustedBatch] = currentScore;

            maxScore = std::max(maxScore, currentScore);
        }

        for (unsigned int currentStep = 0; currentStep < steps; currentStep++)
        {
            unsigned int adjustedBatch = (currentStep + beginOffset) % CONTEXT_WINDOW;
            float currentScore = std::exp(memory.Scores[adjustedBatch] - maxScore);
            memory.Scores[adjustedBatch] = currentScore;

            totalScore += currentScore;
        }

        for (unsigned int currentStep = 0; currentStep < steps; currentStep++)
        {
            unsigned int adjustedBatch = (currentStep + beginOffset) % CONTEXT_WINDOW;
            memory.Scores[adjustedBatch] = memory.Scores[adjustedBatch] / totalScore;
        }


        //compute (qk) * v
        for (unsigned int c = 0; c < 64; c++)
        {
            x.SetValue(0, qhead, c, 0);
        }

        for (unsigned int currentStep = 0; currentStep < steps; currentStep++)
        {
            unsigned adjustedBatch = (currentStep + beginOffset) % CONTEXT_WINDOW;

            for (unsigned int c = 0; c < 64; c++)
            {
                float v = memory.VCache.GetValue(adjustedBatch, kvhead, c) * memory.Scores[adjustedBatch];
                float t = x.GetValue(0, qhead, c);

                x.SetValue(0, qhead, c, t + v);
            }
        }
    }

    x.Reshape(1, 1, 960);
}
 * 
*/


void TensorUtils::GQA(Tensor4D& Q, Tensor4D& K, Tensor4D& V, Tensor4D& scoresTemp, Tensor4D& x, unsigned int position, unsigned int QHeads, unsigned int KVHeads)
{
    
}

void TensorUtils::SiLU(Tensor4D& tgt)
{
    Shape s = tgt.GetShape();
    float* data = tgt.GetStorage();
    unsigned int len = s.B0 * s.B0 * s.L * s.C;

    for (unsigned int i = 0; i < len; i++)
    {
        float val = data[i];
        data[i] = val * (1 / (1 + std::exp(-val)));
    }
}

void TensorUtils::RMSNorm(Tensor4D& tgt, Tensor4D& scales)
{
    if(tgt.IsCacheFriendly())
        throw std::runtime_error("Invalid memory layout.");

    Shape s = tgt.GetShape();
    unsigned int channelSizeBytes = s.C * sizeof(float);

    if(s.C != scales.GetShape().C)
        throw std::runtime_error("Invalid shape");


    for(unsigned int b0 = 0; b0 < s.B0; b0++)
    {
        for(unsigned int b1 = 0; b1 < s.B1; b1++)
        {
            for(unsigned int l = 0; l < s.L; l++)
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