#pragma once

#include "LayerMemory.hpp"
#include "PositionMemory.hpp"

class ModelContext
{
public:
    unsigned int Position = 0;
    PositionMemory PosMem;
    LayerMemory LayerMem[32];

    float Logits[49152];
    Tensor InputX{1, 1, 960};
};