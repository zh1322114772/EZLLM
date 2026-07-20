#include <iostream>
#include "ModelContext.hpp"
#include "Model.hpp"
#include "Tokenizer\GPT2Tokenizer.hpp"
#include "UTF8.hpp"
#include "Tensor4D\Tensor4D.hpp"
#include <chrono>

std::string getFormattedTemplate(std::string role, std::string content) 
{
    return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
}

int main() 
{
    //set utf-8 console
    InitializeUtf8Console();

    Tensor4D projUp(1, 1, 2560, 960);
    Tensor4D projdown(1, 1, 960, 2560);
    Tensor4D res(1, 1, 2560, 2560);
    projUp.FromFile(EZLLM_PROJECT_DIR + std::string("/Model/model.layers.0.mlp.up_proj.weight"));
    projdown.FromFile(EZLLM_PROJECT_DIR + std::string("/Model/model.layers.0.mlp.down_proj.weight"));

    Tensor projUp1(1, 2560, 960);
    Tensor projdown1(1, 960, 2560);
    Tensor res1(1, 2560, 2560);
    projUp1.ReadFromFile(EZLLM_PROJECT_DIR + std::string("/Model/model.layers.0.mlp.up_proj.weight"));
    projdown1.ReadFromFile(EZLLM_PROJECT_DIR + std::string("/Model/model.layers.0.mlp.down_proj.weight"));

    auto start_time = std::chrono::high_resolution_clock::now();
    res1.MatMul(projUp1, projdown1);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = (end_time - start_time) / std::chrono::milliseconds(1);
    std::cout << res1 << std::endl;
    std::cout << time << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    projUp.SetCacheFriendly(true);
    projdown.SetCacheFriendly(true);
    res.MatMul(projUp, projdown);
    res.SetCacheFriendly(false);
    projUp.SetCacheFriendly(false);
    projdown.SetCacheFriendly(false);
    end_time = std::chrono::high_resolution_clock::now();
    time = (end_time - start_time) / std::chrono::milliseconds(1);
    std::cout << res << std::endl;
    std::cout << time << std::endl;

    /***
    ModelContext *context = new ModelContext();
    Model *model = new Model(EZLLM_PROJECT_DIR + std::string("/Model/"));
    GPT2Tokenizer tokenizer(EZLLM_PROJECT_DIR + std::string("/Model/tokenizer/tokenizer.json"));

    const auto eosId = tokenizer.TokenToId("<|im_end|>");
    if (!eosId)
    {
        throw std::runtime_error("Tokenizer has no <|im_end|> token");
    }

    std::string sys_prompt = getFormattedTemplate("system", "you are a helpful assistant");
    model->Generate(tokenizer.Encode(sys_prompt), context, eosId.value(), 1.0f, 0, false);


    while(true)
    {
        std::cout << "[User]: " << std::flush;
        std::string rawInput;
    
        if (!std::getline(std::cin, rawInput))
        {
            std::cout << '\n';
            break;
        }

        // Prevent processing if the user just hits enter without typing
        if (rawInput.empty()) continue; 


        std::string formattedInput = getFormattedTemplate("user", rawInput);
        formattedInput.append("<|im_start|>assistant\n");

        std::vector<int> generatedIds = model->Generate(tokenizer.Encode(formattedInput), context, eosId.value(), 0.6f, 1024, false);

        std::cout << "[Assistant]: "<<tokenizer.Decode(generatedIds) << std::endl;
    }
    */
    return 0;
}