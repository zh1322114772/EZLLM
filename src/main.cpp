#include <iostream>
#include "ModelContext.hpp"
#include "Model.hpp"
#include "Tokenizer\GPT2Tokenizer.hpp"
#include "UTF8.hpp"

std::string getFormattedTemplate(std::string role, std::string content) 
{
    return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
}

int main() 
{
    //set utf-8 console
    InitializeUtf8Console();


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

    return 0;
}