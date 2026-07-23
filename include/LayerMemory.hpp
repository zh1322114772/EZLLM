#pragma once

#include "Tensor.hpp"
#include "Tensor4D/Tensor4D.hpp"
#include "Global.hpp"
struct LayerMemory
{
    public:
        float Scores[CONTEXT_WINDOW];

        Tensor TempX{1, 1, 960};
        Tensor TempProjUp{1, 1, 2560};
        Tensor TempProjGate{1, 1, 2560};

        Tensor Query{1, 1, 960};
        Tensor Key{1, 1, 320};
        Tensor Value{1, 1, 320};

        Tensor KCache{CONTEXT_WINDOW, 5, 64};
        Tensor VCache{CONTEXT_WINDOW, 5, 64};
};