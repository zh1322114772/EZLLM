#pragma once

#include <string>
#include <iostream>

struct TensorShape {
    unsigned int batch;
    unsigned int length;
    unsigned int channels;
};

class Tensor {
    private:
        float* __data = nullptr;
        TensorShape __shape;

        unsigned int GetIndex(unsigned int batch, unsigned int length, unsigned int channel, TensorShape& shape);

        void MatMul(Tensor &m0, Tensor &m1, unsigned int& batchM0, unsigned int& batchM1, unsigned int& batchM2);

    public:
        Tensor(unsigned int batch, unsigned int length, unsigned int channels);

        void CopyFrom(Tensor &t);

        void BatchCopy(Tensor &m0, unsigned int srcBatch, unsigned int tgtBatch);

        void Reshape(unsigned int newBatch, unsigned int newLength, unsigned int newChannels);

        void ReadFromFile(const std::string& filename);

        float GetValue(unsigned int batch, unsigned int length, unsigned int channel);

        void CopyChannels(Tensor &m0, unsigned int batch, unsigned int length);

        void SetValue(unsigned int batch, unsigned int length, unsigned int channel, float value);

        TensorShape GetShape() const;

        void MatMul(Tensor& m0, Tensor& m1);

        void ElementwiseMul(Tensor &m0);

        void Add(Tensor& m0);

        void Add(Tensor& m0, unsigned int b);

        void Add(Tensor& m0, unsigned int b, unsigned int l);

        void Set(float v);

        void T();

        void RMSNorm(Tensor& scale);

        void SiLU();

        friend std::ostream& operator<<(std::ostream& os, Tensor& tensor);

        ~Tensor();
};