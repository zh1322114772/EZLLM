#include "Tensor4D\SlicedTensor4D.hpp"
#include <stdexcept>

SlicedTensor4D::SlicedTensor4D(Shape low, Shape up, Tensor4D *srcTensor):
_lower(low), 
_upper(up)
{
    _tensor = srcTensor;

    //check if valid sliced tensor
    Shape tsShape = _tensor->GetShape();

    if((_lower.B0 < 0) || (_lower.B1 < 0) || (_lower.L < 0) || (_lower.C < 0))
        throw std::invalid_argument("Invalid sliced shape.");

    if((_upper.B0 > tsShape.B0) || (_upper.B1 > tsShape.B1) || (_upper.L > tsShape.L) || (_upper.C > tsShape.C))
        throw std::invalid_argument("Invalid sliced shape.");

    if(((_upper.B0 - _lower.B0) < 1) || ((_upper.B1 - _lower.B1) < 1) || ((_upper.L - _lower.L) < 1) || ((_upper.C - _lower.C) < 1))
        throw std::invalid_argument("Invalid sliced shape.");
}

void SlicedTensor4D::CopyFrom(SlicedTensor4D& src)
{
    // to be implemeneted;
}