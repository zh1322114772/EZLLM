#include "Tensor.hpp"
#include <stdexcept>
#include <fstream>
#include <math.h>

Tensor::Tensor(unsigned int batch, unsigned int length, unsigned int channels) 
{
    __shape.batch = batch;
    __shape.length = length;
    __shape.channels = channels;

    if((__shape.batch * __shape.length * __shape.channels) <= 0)
    {
        throw std::runtime_error("Tensor size cannot be zero.");
    }

    unsigned int size =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(length) *
        static_cast<std::size_t>(channels);

    __data = new float[size];
}

Tensor::~Tensor() 
{
    delete[] __data;
}

TensorShape Tensor::GetShape() const 
{
    return __shape;
}

void Tensor::ReadFromFile(const std::string& filename)
{
    std::cout << "Loading " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    const std::streamsize fileSize = file.tellg();

    if (fileSize < 0) {
        throw std::runtime_error("Could not determine file size: " + filename);
    }

    const std::size_t elementCount =
        static_cast<std::size_t>(__shape.batch) *
        static_cast<std::size_t>(__shape.length) *
        static_cast<std::size_t>(__shape.channels);

    const std::size_t expectedBytes = elementCount * sizeof(float);

    if (static_cast<std::size_t>(fileSize) != expectedBytes) {
        throw std::runtime_error(
            "File size does not match tensor shape. File: " +
            std::to_string(fileSize) +
            " bytes, expected: " +
            std::to_string(expectedBytes) +
            " bytes."
        );
    }

    if (__data == nullptr) {
        throw std::runtime_error("Tensor data pointer is null.");
    }

    file.seekg(0, std::ios::beg);

    if (!file) {
        throw std::runtime_error("Failed to seek to beginning of file.");
    }

    file.read(
        reinterpret_cast<char*>(__data),
        static_cast<std::streamsize>(expectedBytes)
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read tensor file. Read " +
            std::to_string(file.gcount()) +
            " of " +
            std::to_string(expectedBytes) +
            " bytes."
        );
    }

    

    std::cout << "Loaded " << filename << " "
              << expectedBytes
              << " bytes into " << static_cast<void*>(__data)
              << std::endl;
}

void Tensor::Add(Tensor& m0)
{

    for (unsigned int b = 0; b < __shape.batch; b++)
    {
        Add(m0, b);
    }
}

void Tensor::Add(Tensor& m0, unsigned int b)
{
    for (unsigned int l = 0; l < __shape.length; l++)
    {
        Add(m0, b, l);
    }
}

void Tensor::Add(Tensor& m0, unsigned int b, unsigned int l)
{
    for (unsigned int c = 0; c < __shape.channels; c++)
    {
        SetValue(b, l, c, GetValue(b, l, c) + m0.GetValue(b, l, c));
    }
}

void Tensor::RMSNorm(Tensor& scale)
{
    for (unsigned int b = 0; b < __shape.batch; b++)
    {
        for (unsigned int l = 0; l < __shape.length; l++)
        {
            float rms = 0;

            for (unsigned int c = 0; c < __shape.channels; c++)
            {
                float val = GetValue(b, l, c);
                rms = rms + (val * val);
            }

            rms += 1.0E-9;
            rms /= __shape.channels;
            rms = std::sqrt(rms);

            for (unsigned int c = 0; c < __shape.channels; c++)
            {
                float val = GetValue(b, l, c) / rms;
                SetValue(b, l, c, scale.GetValue(0, 0, c) * val);
            }
        }
    }
}

unsigned int Tensor::GetIndex(unsigned int batch, unsigned int length, unsigned int channel, TensorShape& shape)
{
    return (batch * shape.length * shape.channels) + (length * shape.channels) + channel;
}

float Tensor::GetValue(unsigned int batch, unsigned int length, unsigned int channel)
{
    return __data[GetIndex(batch, length, channel, __shape)];
}

void Tensor::SetValue(unsigned int batch, unsigned int length, unsigned int channel, float value) 
{
    __data[GetIndex(batch, length, channel, __shape)] = value;
}

void Tensor::MatMul(Tensor &m0, Tensor &m1, unsigned int& batchM0, unsigned int& batchM1, unsigned int& batchM2)
{
    for (unsigned int l = 0; l < m0.__shape.length; l++)
    {
        for (unsigned int c = 0; c < m1.__shape.channels; c++)
        {
            float sum = 0;

            for (unsigned int i = 0; i < m0.__shape.channels; i++)
            {
                sum += m0.GetValue(batchM0, l, i) * m1.GetValue(batchM1, i, c);
            }

            SetValue(batchM2, l, c, sum);
        }
    }
}

void Tensor::ElementwiseMul(Tensor &m0)
{
    unsigned int len = __shape.batch * __shape.length * __shape.channels;

    for (unsigned int i = 0; i < len; i++)
    {
        __data[i] = __data[i] * m0.__data[i];
    }
}

void Tensor::SiLU()
{
    unsigned int len = __shape.batch * __shape.length * __shape.channels;

    for (unsigned int i = 0; i < len; i++)
    {
        float val = __data[i];
        __data[i] = val * (1 / (1 + std::exp(-val)));
    }
}

void Tensor::MatMul(Tensor& m0, Tensor& m1) 
{
    unsigned int targetBatch = 0;

    if(m0.__shape.batch == 1 || m1.__shape.batch == 1)
    {
        //broadcast
        targetBatch = m0.__shape.batch * m1.__shape.batch;
    }else
    {
        targetBatch = m0.__shape.batch;
    }

    unsigned int m0BatchInc = (m0.__shape.batch != 1);
    unsigned int m1BatchInc = (m1.__shape.batch != 1);
    unsigned int m0CurrentBatch = 0;
    unsigned int m1CurrentBatch = 0;

    for (unsigned int currentBatch = 0; currentBatch < targetBatch; currentBatch++)
    {
        MatMul(m0, m1, m0CurrentBatch, m1CurrentBatch, currentBatch);

        m0CurrentBatch += m0BatchInc;
        m1CurrentBatch += m1BatchInc;
    }
}

void Tensor::CopyFrom(Tensor& t)
{
    std::memcpy(__data, t.__data, (__shape.batch * __shape.length * __shape.channels) * sizeof(float));
    __shape = t.__shape;
}

void Tensor::T() 
{
    float* transposedData = new float[__shape.batch * __shape.length * __shape.channels];
    TensorShape newShape;

    newShape.batch = __shape.batch;
    newShape.length = __shape.channels;
    newShape.channels = __shape.length;

    for (unsigned int b = 0; b < __shape.batch; ++b) 
    {
        for (unsigned int l = 0; l < __shape.length; ++l) 
        {
            for (unsigned int c = 0; c < __shape.channels; ++c) 
            {
                unsigned int originalIndex = GetIndex(b, l, c, __shape);
                unsigned int transposedIndex = GetIndex(b, c, l, newShape);
                transposedData[transposedIndex] = __data[originalIndex];
            }
        }
    }

    delete[] __data;

    __data = transposedData;
    __shape = newShape;
}

void Tensor::BatchCopy(Tensor &m0, unsigned int srcBatch, unsigned int tgtBatch)
{
    if((srcBatch >= m0.__shape.batch) || (tgtBatch >= __shape.batch))
    {
        throw std::runtime_error("Invalid src/tgt batch.");
    }

    if((__shape.length != m0.__shape.length) || (__shape.channels != m0.__shape.channels))
    {
        throw std::runtime_error("Invalid Length/Channels size");
    }

    unsigned bytesToCopy = __shape.length * __shape.channels * sizeof(float);
    float *srcAddr = m0.__data + m0.GetIndex(srcBatch, 0, 0, m0.__shape);
    float *destAddr = __data + GetIndex(tgtBatch, 0, 0, __shape);

    memcpy(destAddr, srcAddr, bytesToCopy);
}

void Tensor::Reshape(unsigned int newBatch, unsigned int newLength, unsigned int newChannels)
{
    __shape.batch = newBatch;
    __shape.length = newLength;
    __shape.channels = newChannels;
}

std::ostream& operator<<(std::ostream& os, Tensor& tensor) 
{
    os << "Tensor Shape: (batch: " << tensor.__shape.batch 
       << ", length: " << tensor.__shape.length 
       << ", channels: " << tensor.__shape.channels << ")";


    for (int b = 0; (b < tensor.__shape.batch) && (b < 5); ++b) {
        os << "\nBatch " << b << ":\n";
        for (int l = 0; (l < tensor.__shape.length) && (l < 5); ++l) {
            os << "Length " << l << ": ";
            for (int c = 0; (c < tensor.__shape.channels) && (c < 20); ++c) {
                os << tensor.GetValue(b, l, c) << " ";
            }
            os << "\n";
        }
    }

    return os;
}

void Tensor::Set(float v)
{
    float len = __shape.batch * __shape.length * __shape.channels;

    for (int i = 0; i < len; i++)
    {
        __data[i] = v;
    }
}

void Tensor::CopyChannels(Tensor &m0, unsigned int batch, unsigned int length)
{
    unsigned int offset = GetIndex(batch, length, 0, m0.__shape);
    unsigned int bytes = m0.__shape.channels * sizeof(float);

    memcpy(__data, &m0.__data[offset], bytes);
}