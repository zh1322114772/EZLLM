
#pragma once
#include <string>

struct Shape
{
    unsigned short int B0;
    unsigned short int B1;
    unsigned short int L;
    unsigned short int C;
};

class Tensor4D
{
private:
    Shape _shape;
    bool _cacheFriendly = false;
    float* _data;
    float*_buffer;

    void Transpose(float *src, float *dst, unsigned short int l, unsigned short int c);

    void MatMul88(float *m0, float *m1, float *dest, unsigned int s0,  unsigned int s1,  unsigned int s2);

    //void BatchMul88(Tensor4D &m0, Tensor4D &m1, unsigned int offsetm0, unsigned int offsetm1);
    //to be implemeneted

    void CacheFriendly(float *src, float *dst, unsigned short int l, unsigned short int c);

    void AccessFriendly(float *src, float *dst, unsigned short int l, unsigned short int c);

public:
    Tensor4D(unsigned short int B0, unsigned short int B1, unsigned short int L, unsigned short int C);

    ~Tensor4D();

    void T();

    void Reshape(unsigned short int B0, unsigned short int B1, unsigned short int L, unsigned short int C);

    void FromFile(const std::string& path);

    void SetCacheFriendly(bool flag);

    void Set(float val);

    void MatMul(Tensor4D &m0, Tensor4D &m1);

    friend std::ostream& operator<<(std::ostream& os, Tensor4D& tensor);
};
