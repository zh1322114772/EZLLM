
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

    unsigned int GetIndex(unsigned short int b0, unsigned short int b1, unsigned short int l, unsigned short int c);

    void ComputeOffsets();

public:
    Tensor4D(unsigned short int B0, unsigned short int B1, unsigned short int L, unsigned short int C);

    ~Tensor4D();

    void T();

    void Reshape(unsigned short int B0, unsigned short int B1, unsigned short int L, unsigned short int C);

    Shape GetShape();

    void FromFile(const std::string& path);

    void SetCacheFriendly(bool flag);

    bool IsCacheFriendly();

    void Add(Tensor4D& src);

    void Set(float val);

    void MatMul(Tensor4D &m0, Tensor4D &m1);

    float GetValue(unsigned short int b0, unsigned short int b1, unsigned short int l, unsigned short int c);

    void SetValue(unsigned short int b0, unsigned short int b1, unsigned short int l, unsigned short int c, float v);

    SlicedTensor4D AsSlicedTensor(
        unsigned short int b0l,
        unsigned short int b0u,
        unsigned short int b1l,
        unsigned short int b1u,
        unsigned short int ll,
        unsigned short int lu,
        unsigned short int cl,
        unsigned short int cu);

    

    float *GetStorage();

    float *GetStorage(unsigned short int b0, unsigned short int b1, unsigned short int l, unsigned short int c);

    float *GetLCBuffer();

    friend std::ostream& operator<<(std::ostream& os, Tensor4D& tensor);
};
