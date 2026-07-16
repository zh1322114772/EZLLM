#include "Tokenizer/Tokenizer.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

Tokenizer::Tokenizer(std::string path)
{
    std::ifstream fileStream(path);
    json data = json::parse(fileStream);
}
