#pragma once
#include "Tensor4D/Tensor4D.hpp"

class SlicedTensor4D
{
private:
    Shape _lower;
    Shape _upper;
    Tensor4D* _tensor;
public:
    SlicedTensor4D(Shape low, Shape up, Tensor4D *srcTensor);

    void CopyFrom(SlicedTensor4D &src);
};