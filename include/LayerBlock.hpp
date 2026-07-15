#pragma once

#include<string>
#include "LayerMemory.hpp"
#include "PositionMemory.hpp"
#include "Tensor.hpp"

class LayerBlock 
{
    private:
        Tensor __q, __k, __v, __o;
        Tensor __inputNorm, __postNorm;
        Tensor __upProj, __gateProj, __downProj;

        void Attention(LayerMemory &memory, PositionMemory &pmemory, Tensor &x, unsigned int position);

        void RoPE(Tensor &x, PositionMemory &pmemory);

        void GroupedQueryAttention(LayerMemory &memory, Tensor &q, Tensor &x, unsigned int position);

        void SwiGLU(LayerMemory &memory, Tensor &x);

    public:
        LayerBlock(std::string layerPath);

        void Generate(Tensor &x, LayerMemory &memory, PositionMemory &pmemory, unsigned int position);
};