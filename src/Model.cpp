#include "Model.hpp"
#include "TensorUtils.hpp"
#include <limits>
#include <math.h>

std::vector<int> Model::Generate(std::vector<int> inputTokens, ModelContext *context, int eos, float temperture, int maxTokens, bool sample)
{
    int last = inputTokens.back();
    inputTokens.pop_back();

    //pre-fill
    if(inputTokens.size() > 0)
    {
        Prefill(inputTokens, context);
    }

    //generate new tokens
    std::vector<int> ret;

    for (int i = 0; i < maxTokens; i++)
    {
        int current = Next(last, context, temperture, sample);
        ret.push_back(current);

        if(current == eos)
            break;

        last = current;
    }

    return ret;
}

void Model::Prefill(std::vector<int> tokens, ModelContext *context)
{
    for (int i = 0; i < tokens.size(); i++)
    {
        Iter(context, tokens[i]);
    }
    
}

void Model::Iter(ModelContext *context, int token)
{
    //get position embedding
    TensorUtils::ComputeRoPEPositions(context->PosMem, context->Position);

    //get token embedding
    context->InputX.CopyChannels(*_embedding, 0, token);

    //run through transformer blocks
    for (int i = 0; i < 32; i++)
    {
        _transBlocks[i]->Generate(context->InputX, context->LayerMem[i], context->PosMem, context->Position);
    }

    context->Position += 1;
}

int Model::Next(int lastToken, ModelContext *context, float temperture, bool sample)
{
    Iter(context, lastToken);

    //final norm
    context->InputX.RMSNorm(*_finalNorm);

    //compute Logits
    float maxScore = std::numeric_limits<float>::lowest();
    int maxId = 0;
    for (int i = 0; i < 49152; i++)
    {
        float score = TensorUtils::ChannelDot(*_embedding, context->InputX, 0, i, 0, 0);
        context->Logits[i] = score;

        if(score > maxScore)
        {
            maxScore = score;
            maxId = i;
        }

    }

    if(!sample)
        return maxId;

    //for numerical stability & temperture
    float all = 0;
    for (int i = 0; i < 49152; i++)
    {
        float score = exp((context->Logits[i] - maxScore) / temperture);
        context->Logits[i] = score;
        all += score;
    }

    //normalize
    for (int i = 0; i < 49152; i++)
    {
        context->Logits[i] /= all;
    }

    //sample
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    float randomNum = dis(_gen);
    float cumulativeProbability = 0.0f;
    int selectedId = 49151;

    for (int i = 0; i < 49152; i++)
    {
        cumulativeProbability += context->Logits[i];

        if (randomNum < cumulativeProbability)
        {
            selectedId = i;
            break;
        }
    }

    return selectedId;
}

Model::Model(std::string modelPath): _gen(_rd())
{
    _embedding = new Tensor(1, 49152, 960);
    _finalNorm = new Tensor(1, 1, 960);

    _embedding->ReadFromFile(modelPath + "model.embed_tokens.weight");
    _finalNorm->ReadFromFile(modelPath + "model.norm.weight");

    for (int i = 0; i < 32; i++)
    {
        _transBlocks[i] = new LayerBlock(modelPath + "model.layers." + std::to_string(i));
    }
}

Model::~Model()
{
    delete _embedding;
    delete _finalNorm;

    for (int i = 0; i < 32; i++)
    {
        delete _transBlocks[i];
    }
}