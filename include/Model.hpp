#include "LayerBlock.hpp"
#include "ModelContext.hpp"
#include <vector>
#include <random>

class Model
{
private:
    Tensor *_embedding = nullptr;
    LayerBlock* _transBlocks[32];
    Tensor *_finalNorm = nullptr;
    std::random_device _rd; 
    std::mt19937 _gen; 

    void Prefill(std::vector<int> tokens, ModelContext *context);
    int Next(int lastToken, ModelContext *context, float temperture, bool sample);

    void Iter(ModelContext *context, int token);

public:
    ~Model();
    Model(std::string modelPath);

    std::vector<int> Generate(std::vector<int> inputTokens, ModelContext *context, int eos, float temperture, int maxTokens, bool sample);
};