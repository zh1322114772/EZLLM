#include "Tensor4D\Tensor4D.hpp"
#include "Tensor4D\MathOps.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <math.h>



Tensor4D::Tensor4D(unsigned short int B0, unsigned short int B1, unsigned short int L, unsigned short int C)
{
    _shape.B0 = B0;
    _shape.B1 = B1;
    _shape.L = L;
    _shape.C = C;

    if ((B0 * B1 * L * C) <= 0)
        throw std::invalid_argument("Invalid shape");

    _data = new float[B0 * B1 * L * C];
    _buffer = new float[L * C];
}

void Tensor4D::T()
{
    if(_cacheFriendly)
        throw std::runtime_error("Must set cache friendly to false before transpose.");

    unsigned int matSize = _shape.L * _shape.C;

    for (int b0 = 0; b0 < _shape.B0; b0++)
    {
        for (int b1 = 0; b0 < _shape.B1; b1++)
        {
            unsigned int offset = (b0 * _shape.B1 * matSize) + (b1 * matSize);
            MathOps::Transpose(_data + offset, _buffer, _shape.L, _shape.C);
            memcpy(_data + offset, _buffer, static_cast<std::size_t>(matSize * sizeof(float)));
        }
    }

    unsigned short int temp = _shape.C;
    _shape.C = _shape.L;
    _shape.L = temp;
}

void Tensor4D::Reshape(unsigned short int B0, unsigned short int B1, unsigned short int L, unsigned short int C)
{
    unsigned int n = B0 * B1 * L * C;
    unsigned int c = _shape.B0 * _shape.B1 * _shape.L * _shape.C;

    if((n <= 0) || (n != c))
        throw std::invalid_argument("Invalid Shape");

    if((_shape.L * _shape.C) != (L * C))
    {
        float *newBuff = new float[L * C];
        delete[] _buffer;
        _buffer = newBuff;
    }

    _shape.B0 = B0;
    _shape.B1 = B1;
    _shape.L = L;
    _shape.C = C;
}


void Tensor4D::FromFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    const std::streamsize fileSize = file.tellg();

    if (fileSize < 0) {
        throw std::runtime_error("Could not determine file size: " + path);
    }

    const std::size_t elementCount = static_cast<std::size_t>(_shape.B0 * _shape.B1 * _shape.L * _shape.C);
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

    file.seekg(0, std::ios::beg);

    if (!file) {
        throw std::runtime_error("Failed to seek to beginning of file.");
    }

    file.read(
        reinterpret_cast<char*>(_data),
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

    std::cout << "Loaded " << path << " "
              << expectedBytes
              << " bytes into " << static_cast<void*>(_data)
              << std::endl;
}

void Tensor4D::SetCacheFriendly(bool flag)
{
    if(_cacheFriendly == flag)
        return;

    if((_shape.L % 8) || (_shape.C % 8))
        throw std::runtime_error("Invalid Shape");

    unsigned int matSize = _shape.L * _shape.C;

    for (unsigned short int b0 = 0; b0 < _shape.B0; b0++)
    {
        for (unsigned short int b1 = 0; b1 < _shape.B1; b1++)
        {
            unsigned int offset = (b0 * _shape.B1 * matSize) + (b1 * matSize);

            if(flag)
            {
                MathOps::CacheFriendly(_data + offset, _buffer, _shape.L, _shape.C);
            }else
            {
                MathOps::AccessFriendly(_data + offset, _buffer, _shape.L, _shape.C);
            }

            memcpy(_data + offset, _buffer, matSize * sizeof(float));
        }
    }

    _cacheFriendly = !_cacheFriendly;
}

void Tensor4D::Set(float val)
{
    std::fill(_data, _data + (_shape.B0 * _shape.B1 * _shape.L * _shape.C), val);
}

void Tensor4D::MatMul(Tensor4D& m0, Tensor4D& m1)
{

    if(!m0._cacheFriendly || !m1._cacheFriendly)
        throw std::runtime_error("m0 and m1 must be cache friendly.");

    if(m0._shape.C != m1._shape.L)
        throw std::runtime_error("Invalid shape.");

    if((_shape.L != m0._shape.L) || (_shape.C != m1._shape.C))
        throw std::runtime_error("Invalid shape.");

    unsigned short int m0b0Inc = 0;
    unsigned short int m0b1Inc = 0;
    unsigned short int m1b0Inc = 0;
    unsigned short int m1b1Inc = 0;

    if(m0._shape.B1 == m1._shape.B1)
    {
        m0b1Inc = 1;
        m1b1Inc = 1;
    }else if((m0._shape.B1 == 1) || (m1._shape.B1 == 1))
    {
        m0b1Inc = !(m0._shape.B1 == 1);
        m1b1Inc = !(m1._shape.B1 == 1);
    }else
    {
        throw std::runtime_error("Invalid shape.");
    }

    if(m0._shape.B0 == m1._shape.B0)
    {
        m0b0Inc = 1;
        m1b0Inc = 1;
    }else if((m0._shape.B0 == 1) || (m1._shape.B0 == 1))
    {
        m0b0Inc = !(m0._shape.B0 == 1);
        m1b0Inc = !(m1._shape.B0 == 1);
    }else
    {
        throw std::runtime_error("Invalid shape.");
    }

    unsigned short int newB0 = std::max(m0._shape.B0, m1._shape.B0);
    unsigned short int newB1 = std::max(m0._shape.B1, m1._shape.B1);
    unsigned int m0MatSize = m0._shape.L * m0._shape.C;
    unsigned int m1MatSize = m1._shape.L * m1._shape.C;
    unsigned int tgtMatSize = _shape.L * _shape.C;

    if((newB0 != _shape.B0) || (newB1 != _shape.B1))
        throw std::runtime_error("Invalid shape.");

    if((_shape.L % 8) || (_shape.C % 8))
        throw std::runtime_error("Invalid shape.");

    //set _data to 0
    Set(0);

    unsigned short int m1b0 = 0;
    unsigned short int m0b0 = 0;

    for (unsigned short int b0 = 0; b0 < newB0; b0++)
    {
        unsigned short int m1b1 = 0;
        unsigned short int m0b1 = 0;

        for (unsigned short int b1 = 0; b1 < newB1; b1++)
        {
            
            unsigned int m0Offset = m0b0 * (m0._shape.B1 * m0MatSize) + (m0b1 * m0MatSize);
            unsigned int m1Offset = m1b0 * (m1._shape.B1 * m1MatSize) + (m1b1 * m1MatSize);
            unsigned int tgtOffset = b0 * (_shape.B1 * tgtMatSize) + (b1 * tgtMatSize);
            
            MathOps::MatMul88(m0._data + m0Offset, m1._data + m1Offset, _data + tgtOffset, m0._shape.L, m0._shape.C, m1._shape.C);

            m0b1 += m0b1Inc;
            m1b1 += m1b1Inc;
        }

        m0b0 += m0b0Inc;
        m1b0 += m1b0Inc;
    }
    _cacheFriendly = true;
}

SlicedTensor4D Tensor4D::AsSlicedTensor(
        unsigned short int b0l,
        unsigned short int b0u,
        unsigned short int b1l,
        unsigned short int b1u,
        unsigned short int ll,
        unsigned short int lu,
        unsigned short int cl,
        unsigned short int cu)
{
    Shape lower;
    Shape upper;
    lower.B0 = b0l;
    lower.B1 = b1l;
    lower.L = ll;
    lower.C = cl;
    upper.B0 = b0u;
    upper.B1 = b1u;
    upper.L = lu;
    upper.C = cu;

    return SlicedTensor4D(lower, upper, this);
}

bool Tensor4D::IsCacheFriendly()
{
    return _cacheFriendly;
}

std::ostream& operator<<(std::ostream& os, Tensor4D& tensor)
{
    os << "Tensor Shape: ("
       << tensor._shape.B0 << ", "
       << tensor._shape.B1 << ", "
       << tensor._shape.L  << ", "
       << tensor._shape.C  << ")\n";

    const unsigned int maxB0 = std::min<unsigned int>(tensor._shape.B0, 5);
    const unsigned int maxB1 = std::min<unsigned int>(tensor._shape.B1, 5);
    const unsigned int maxL  = std::min<unsigned int>(tensor._shape.L, 5);
    const unsigned int maxC  = std::min<unsigned int>(tensor._shape.C, 20);

    for (unsigned int b0 = 0; b0 < maxB0; ++b0)
    {
        for (unsigned int b1 = 0; b1 < maxB1; ++b1)
        {
            os << "\nBatch (" << b0 << ", " << b1 << "):\n";

            for (unsigned int l = 0; l < maxL; ++l)
            {
                os << "Length " << l << ": ";

                for (unsigned int c = 0; c < maxC; ++c)
                {
                    const std::size_t index =
                        static_cast<std::size_t>(b0) * tensor._shape.B1 * tensor._shape.L * tensor._shape.C
                        + static_cast<std::size_t>(b1) * tensor._shape.L * tensor._shape.C
                        + static_cast<std::size_t>(l)  * tensor._shape.C
                        + static_cast<std::size_t>(c);

                    os << tensor._data[index] << ' ';
                }

                if (tensor._shape.C > maxC)
                {
                    os << "...";
                }

                os << '\n';
            }

            if (tensor._shape.L > maxL)
            {
                os << "...\n";
            }
        }
    }

    if (tensor._shape.B0 > maxB0 || tensor._shape.B1 > maxB1)
    {
        os << "\n...\n";
    }

    return os;
}

Shape Tensor4D::GetShape()
{
    return _shape;
}

Tensor4D::~Tensor4D()
{
    delete[] _data;
    delete[] _buffer;
}