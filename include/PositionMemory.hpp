#pragma once

#include "Tensor.hpp"
struct PositionMemory
{
    public:
        Tensor sin{1, 1, 32};
        Tensor cos{1, 1, 32};
};