#include <iostream>
#include "ModelContext.hpp"
#include "Model.hpp"
#include "Tokenizer\Tokenizer.hpp"
#include "UTF8.hpp"

int main() {
    //set utf-8 console
    InitializeUtf8Console();


    ModelContext *context = new ModelContext();
    Model *model = new Model(EZLLM_PROJECT_DIR + std::string("/Model/"));
    Tokenizer *tokenizer = new Tokenizer(EZLLM_PROJECT_DIR + std::string("/Model/tokenizer/tokenizer.json"));

    std::vector<int> question = {28120, 28120, 28120};
    std::vector<int> ret = model->Generate(question, context, 2, 0.7, 10, true);

    for (int i = 0; i < ret.size(); i ++)
    {
        std::cout << ret[i] << " " << std::endl;
    }

    return 0;
}