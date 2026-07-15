#include "LayerBlock.hpp"
#include "TensorUtils.hpp"
#include <math.h>
#include <limits>
#include "Global.hpp"

LayerBlock::LayerBlock(std::string layerPath):
    __q{1, 960, 960},
    __k{1, 320, 960},
    __v{1, 320, 960},
    __o{1, 960, 960},
    __inputNorm{1, 1, 960},
    __postNorm{1, 1, 960},
    __upProj{1, 2560, 960},
    __gateProj{1, 2560, 960},
    __downProj{1, 960, 2560}
{
    __q.ReadFromFile(layerPath + ".self_attn.q_proj.weight");
    __k.ReadFromFile(layerPath + ".self_attn.k_proj.weight");
    __v.ReadFromFile(layerPath + ".self_attn.v_proj.weight");
    __o.ReadFromFile(layerPath + ".self_attn.o_proj.weight");
    
    __q.T();
    __k.T();
    __v.T();
    __o.T();


    //norm scales
    __inputNorm.ReadFromFile(layerPath + ".input_layernorm.weight");
    __postNorm.ReadFromFile(layerPath + ".post_attention_layernorm.weight");

    
    //SwiGLU projections
    __upProj.ReadFromFile(layerPath + ".mlp.up_proj.weight");
    __gateProj.ReadFromFile(layerPath + ".mlp.gate_proj.weight");
    __downProj.ReadFromFile(layerPath + ".mlp.down_proj.weight");

    __upProj.T();
    __gateProj.T();
    __downProj.T();
}

void LayerBlock::Generate(Tensor &x, LayerMemory& memory, PositionMemory &pmemory, unsigned int position)
{
    //attention
    memory.TempX.CopyFrom(x);
    Attention(memory, pmemory, memory.TempX, position);

    //residual
    x.Add(memory.TempX);

    //SwiGLU
    memory.TempX.CopyFrom(x);
    SwiGLU(memory, memory.TempX);

    //residual
    x.Add(memory.TempX);
}

void LayerBlock::SwiGLU(LayerMemory &memory, Tensor &x)
{
    //SwiGLU norm
    x.RMSNorm(__postNorm);

    memory.TempProjUp.MatMul(x, __upProj);
    memory.TempProjGate.MatMul(x, __gateProj);
    memory.TempProjGate.SiLU();
    memory.TempProjUp.ElementwiseMul(memory.TempProjGate);
    x.MatMul(memory.TempProjUp, __downProj);
}

void LayerBlock::RoPE(Tensor &x, PositionMemory& pmem)
{
    TensorShape shape = x.GetShape();
    unsigned int half = shape.channels / 2;

    for (unsigned int len = 0; len < shape.length; len++)
    {
        for (unsigned int c = 0; c < half; c++)
        {
            float x0 = x.GetValue(0, len, c);
            float x32 = x.GetValue(0, len, c + half);
            float si = pmem.sin.GetValue(0, 0, c);
            float co = pmem.cos.GetValue(0, 0, c);

            x.SetValue(0, len, c, (x0 * co) - (x32 * si));
            x.SetValue(0, len, c + half, (x32 * co) + (x0 * si));
        }
    }
}

void LayerBlock::GroupedQueryAttention(LayerMemory &memory, Tensor &q, Tensor &x, unsigned int position)
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

void LayerBlock::Attention(LayerMemory& memory, PositionMemory &pmemory, Tensor& x, unsigned int position)
{
    //rms norm
    x.RMSNorm(__inputNorm);

    //self attention
    memory.Query.MatMul(x, __q);
    memory.Key.MatMul(x, __k);
    memory.Value.MatMul(x, __v);

    //reshape Q from (1, 1, 960) to (1, 15, 64), K from (1, 1, 320) to (1, 5, 64) and V from(1, 1, 320) to (1, 5, 64)
    memory.Query.Reshape(1, 15, 64);
    memory.Key.Reshape(1, 5, 64);
    memory.Value.Reshape(1, 5, 64);

    //add Relative position info to Q and K
    RoPE(memory.Query, pmemory);
    RoPE(memory.Key, pmemory);

    //add to KV Cache
    memory.KCache.BatchCopy(memory.Key, 0, position % CONTEXT_WINDOW);
    memory.VCache.BatchCopy(memory.Value, 0, position % CONTEXT_WINDOW);

    //grouped query attention
    GroupedQueryAttention(memory, memory.Query, x, position);

    //reshape back
    memory.Query.Reshape(1, 1, 960);
    memory.Key.Reshape(1, 1, 320);
    memory.Value.Reshape(1, 1, 320);

    // multiply with proj_o
    memory.Query.MatMul(x, __o);
    x.CopyFrom(memory.Query);
}