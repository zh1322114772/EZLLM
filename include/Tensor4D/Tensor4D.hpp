
#pragma once
#include <string>
#include "Tensor4D/Shape.hpp"

class SlicedTensor4D;

class Tensor4D
{
private:
    Shape _shape;
    Shape _offsets;
    bool _cacheFriendly = false;
    float* _data;
    float*_buffer;

    unsigned int GetIndex(unsigned int b0, unsigned int b1, unsigned int l, unsigned int c);

    void ComputeOffsets();

public:
    Tensor4D(unsigned int B0, unsigned int B1, unsigned int L, unsigned int C);

    ~Tensor4D();

    void T();

    void Reshape(unsigned int B0, unsigned int B1, unsigned int L, unsigned int C);

    Shape GetShape();

    void FromFile(const std::string& path);

    void SetCacheFriendly(bool flag);

    bool IsCacheFriendly();

    void Add(Tensor4D& src);

    void Set(float val);

    void MatMul(Tensor4D &m0, Tensor4D &m1);

    float GetValue(unsigned int b0, unsigned int b1, unsigned int l, unsigned int c);

    void SetValue(unsigned int b0, unsigned int b1, unsigned int l, unsigned int c, float v);

    SlicedTensor4D AsSlicedTensor(
        unsigned int b0l,
        unsigned int b0u,
        unsigned int b1l,
        unsigned int b1u,
        unsigned int ll,
        unsigned int lu,
        unsigned int cl,
        unsigned int cu);

    

    float *GetStorage();

    float *GetStorage(unsigned int b0, unsigned int b1, unsigned int l, unsigned int c);

    float *GetLCBuffer();

    friend std::ostream& operator<<(std::ostream& os, Tensor4D& tensor);
};
