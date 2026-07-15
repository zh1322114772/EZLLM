#include <iostream>
#include "ModelContext.hpp"
#include "Model.hpp"

int main() {
    ModelContext *context = new ModelContext();
    Model *model = new Model(EZLLM_PROJECT_DIR + std::string("/Model/"));

    std::vector<int> question = {12, 13};
    std::vector<int> ret = model->Generate(question, context, 2, 0.7, 50);

    for (int i = 0; i < ret.size(); i ++)
    {
        std::cout << ret[i] << " " << std::endl;
    }

    return 0;
}