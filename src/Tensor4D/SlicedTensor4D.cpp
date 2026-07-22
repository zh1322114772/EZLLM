#include "Tensor4D\SlicedTensor4D.hpp"
#include "Tensor4D\Tensor4D.hpp"
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
    if(src._tensor->IsCacheFriendly() || _tensor->IsCacheFriendly())
        throw std::invalid_argument("Invalid cache layout.");

    if(src._tensor->GetStorage() == _tensor->GetStorage())
        throw std::invalid_argument("Invalid src/dst tensors");

        
    unsigned short int b0 = _upper.B0 - _lower.B0;
    unsigned short int b1 = _upper.B1 - _lower.B1;
    unsigned short int l = _upper.L - _lower.L;
    unsigned short int c = _upper.C - _lower.C;

    if((b0 != (src._upper.B0 - src._lower.B0)) || (b1 != (src._upper.B1 - src._lower.B1)) || (l != (src._upper.L - src._lower.L)) || (c != (src._upper.C - src._lower.C)))
        throw std::invalid_argument("Invalid sliced shape.");


    unsigned int numOfBytes = c * sizeof(float);
    Shape dstShape = _tensor->GetShape();
    Shape srcShape = src._tensor->GetShape();

    unsigned int dstOffsetb0 = dstShape.B1 * dstShape.L * dstShape.C;
    unsigned int dstOffsetb1 = dstShape.L * dstShape.C;
    unsigned int dstOffsetl = dstShape.C;
    unsigned int srcOffsetb0 = srcShape.B1 * srcShape.L * srcShape.C;
    unsigned int srcOffsetb1 = srcShape.L * srcShape.C;
    unsigned int srcOffsetl = srcShape.C;

    for (unsigned short int cb0 = 0; cb0 < b0; cb0++)
    {
        for (unsigned short int cb1 = 0; cb1 < b1; cb1++)
        {
            for (unsigned short int cl = 0; cl < l; cl++)
            {
                unsigned int dstOffset = ((cb0 + _lower.B0) * dstOffsetb0) +
                                         ((cb1 + _lower.B1) * dstOffsetb1) +
                                         ((cl + _lower.L) * dstOffsetl) +
                                         _lower.C;

                unsigned int srcOffset = ((cb0 + src._lower.B0) * srcOffsetb0) +
                                         ((cb1 + src._lower.B1) * srcOffsetb1) +
                                         ((cl + src._lower.L) * srcOffsetl) +
                                         src._lower.C;

                memcpy(_tensor->GetStorage() + dstOffset, src._tensor->GetStorage() + srcOffset, numOfBytes);
            }
        }
    }
}