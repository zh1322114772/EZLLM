#pragma once

// Header-only GPT-2 style byte-level BPE tokenizer.
//
// Input format:
//   Hugging Face tokenizer.json containing a BPE model with "vocab" and
//   "merges" fields. Added/special tokens are also loaded when present.
//
// Dependency:
//   nlohmann/json.hpp
//
// Notes:
//   * Designed for GPT-2/SmolLM2-style byte-level BPE tokenizers.
//   * The pre-tokenizer implements the behavior of the classic GPT-2 regex.
//   * Unicode classification is intentionally lightweight and dependency-free.
//     It handles ASCII, CJK, common scripts, punctuation, symbols, and emoji
//     well, but it is not a complete replacement for ICU Unicode properties.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "json.hpp"


class GPT2Tokenizer
{
public:
    using TokenId = int;

    GPT2Tokenizer() = default;

    explicit GPT2Tokenizer(const std::filesystem::path& tokenizerJsonPath)
    {
        Load(tokenizerJsonPath);
    }

    static GPT2Tokenizer FromFile(const std::filesystem::path& tokenizerJsonPath)
    {
        return GPT2Tokenizer(tokenizerJsonPath);
    }

    void Load(const std::filesystem::path& tokenizerJsonPath)
    {
        std::ifstream input(tokenizerJsonPath, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error(
                "Unable to open tokenizer JSON: " + tokenizerJsonPath.string());
        }

        nlohmann::json root;
        input >> root;

        if (!root.contains("model") || !root["model"].is_object())
        {
            throw std::runtime_error("tokenizer.json does not contain a model object");
        }

        const auto& model = root["model"];
        if (model.value("type", std::string{}) != "BPE")
        {
            throw std::runtime_error("GPT2Tokenizer requires a BPE tokenizer model");
        }

        Clear();
        BuildByteTables();
        LoadVocabulary(model);
        LoadMerges(model);
        LoadAddedTokens(root);

        if (model.contains("unk_token") && !model["unk_token"].is_null())
        {
            const std::string unknownToken = model["unk_token"].get<std::string>();
            const auto found = encoder_.find(unknownToken);
            if (found != encoder_.end())
            {
                unknownTokenId_ = found->second;
            }
        }

        loaded_ = true;
    }

    [[nodiscard]] std::vector<TokenId> Encode(std::string_view text) const
    {
        EnsureLoaded();

        std::vector<TokenId> output;
        EncodeRange(text, 0, text.size(), output);
        return output;
    }

    [[nodiscard]] std::string Decode(
        const std::vector<TokenId>& tokenIds,
        bool skipSpecialTokens = false) const
    {
        EnsureLoaded();

        std::string output;
        std::string byteMappedText;

        auto flushByteMappedText = [&]()
        {
            if (byteMappedText.empty())
            {
                return;
            }

            DecodeByteMappedText(byteMappedText, output);
            byteMappedText.clear();
        };

        for (const TokenId tokenId : tokenIds)
        {
            if (tokenId < 0 ||
                static_cast<std::size_t>(tokenId) >= decoder_.size() ||
                !decoder_[static_cast<std::size_t>(tokenId)].has_value())
            {
                throw std::out_of_range("Token ID is outside the vocabulary");
            }

            const auto special = specialTokenIds_.find(tokenId);
            if (special != specialTokenIds_.end())
            {
                flushByteMappedText();
                if (!skipSpecialTokens)
                {
                    output.append(*decoder_[static_cast<std::size_t>(tokenId)]);
                }
            }
            else
            {
                byteMappedText.append(*decoder_[static_cast<std::size_t>(tokenId)]);
            }
        }

        flushByteMappedText();
        return output;
    }

    [[nodiscard]] std::optional<TokenId> TokenToId(std::string_view token) const
    {
        EnsureLoaded();
        const auto found = encoder_.find(std::string(token));
        if (found == encoder_.end())
        {
            return std::nullopt;
        }
        return found->second;
    }

    [[nodiscard]] std::optional<std::string_view> IdToToken(TokenId tokenId) const
    {
        EnsureLoaded();
        if (tokenId < 0 ||
            static_cast<std::size_t>(tokenId) >= decoder_.size() ||
            !decoder_[static_cast<std::size_t>(tokenId)].has_value())
        {
            return std::nullopt;
        }

        return std::string_view(*decoder_[static_cast<std::size_t>(tokenId)]);
    }

    [[nodiscard]] std::size_t VocabularySize() const noexcept
    {
        return encoder_.size();
    }

private:
    struct PairHash
    {
        std::size_t operator()(const std::pair<std::string, std::string>& pair) const noexcept
        {
            const std::size_t firstHash = std::hash<std::string>{}(pair.first);
            const std::size_t secondHash = std::hash<std::string>{}(pair.second);
            return firstHash ^ (secondHash + 0x9e3779b97f4a7c15ULL +
                                (firstHash << 6U) + (firstHash >> 2U));
        }
    };

    struct Utf8Unit
    {
        char32_t codePoint{};
        std::size_t offset{};
        std::size_t length{};
    };

    struct AddedToken
    {
        std::string content;
        TokenId id{};
        bool special{};
    };

    std::unordered_map<std::string, TokenId> encoder_;
    std::vector<std::optional<std::string>> decoder_;
    std::unordered_map<std::pair<std::string, std::string>, std::size_t, PairHash>
        mergeRanks_;

    std::array<std::string, 256> byteEncoder_{};
    std::unordered_map<std::string, std::uint8_t> byteDecoder_;

    std::vector<AddedToken> addedTokens_;
    std::unordered_set<TokenId> specialTokenIds_;
    std::optional<TokenId> unknownTokenId_;
    bool loaded_ = false;

    void Clear()
    {
        encoder_.clear();
        decoder_.clear();
        mergeRanks_.clear();
        byteDecoder_.clear();
        addedTokens_.clear();
        specialTokenIds_.clear();
        unknownTokenId_.reset();
        loaded_ = false;
    }

    void EnsureLoaded() const
    {
        if (!loaded_)
        {
            throw std::logic_error("Tokenizer has not been loaded");
        }
    }

    static void AppendUtf8(char32_t codePoint, std::string& output)
    {
        if (codePoint <= 0x7F)
        {
            output.push_back(static_cast<char>(codePoint));
        }
        else if (codePoint <= 0x7FF)
        {
            output.push_back(static_cast<char>(0xC0U | (codePoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
        }
        else if (codePoint <= 0xFFFF)
        {
            if (codePoint >= 0xD800 && codePoint <= 0xDFFF)
            {
                throw std::runtime_error("UTF-8 cannot encode a surrogate code point");
            }

            output.push_back(static_cast<char>(0xE0U | (codePoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
        }
        else if (codePoint <= 0x10FFFF)
        {
            output.push_back(static_cast<char>(0xF0U | (codePoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codePoint >> 12U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
        }
        else
        {
            throw std::runtime_error("Unicode code point is out of range");
        }
    }

    static Utf8Unit ReadUtf8Unit(std::string_view text, std::size_t offset)
    {
        if (offset >= text.size())
        {
            throw std::out_of_range("UTF-8 offset is out of range");
        }

        const auto first = static_cast<std::uint8_t>(text[offset]);
        Utf8Unit unit{};
        unit.offset = offset;

        if (first <= 0x7F)
        {
            unit.codePoint = first;
            unit.length = 1;
            return unit;
        }

        int continuationCount = 0;
        char32_t codePoint = 0;
        char32_t minimumCodePoint = 0;

        if ((first & 0xE0U) == 0xC0U)
        {
            continuationCount = 1;
            codePoint = first & 0x1FU;
            minimumCodePoint = 0x80;
        }
        else if ((first & 0xF0U) == 0xE0U)
        {
            continuationCount = 2;
            codePoint = first & 0x0FU;
            minimumCodePoint = 0x800;
        }
        else if ((first & 0xF8U) == 0xF0U)
        {
            continuationCount = 3;
            codePoint = first & 0x07U;
            minimumCodePoint = 0x10000;
        }
        else
        {
            throw std::runtime_error("Invalid UTF-8 leading byte");
        }

        if (offset + static_cast<std::size_t>(continuationCount) >= text.size())
        {
            throw std::runtime_error("Truncated UTF-8 sequence");
        }

        for (int index = 1; index <= continuationCount; ++index)
        {
            const auto next = static_cast<std::uint8_t>(text[offset + index]);
            if ((next & 0xC0U) != 0x80U)
            {
                throw std::runtime_error("Invalid UTF-8 continuation byte");
            }
            codePoint = static_cast<char32_t>((codePoint << 6U) | (next & 0x3FU));
        }

        if (codePoint < minimumCodePoint || codePoint > 0x10FFFF ||
            (codePoint >= 0xD800 && codePoint <= 0xDFFF))
        {
            throw std::runtime_error("Invalid UTF-8 code point");
        }

        unit.codePoint = codePoint;
        unit.length = static_cast<std::size_t>(continuationCount + 1);
        return unit;
    }

    static std::vector<Utf8Unit> SplitUtf8Units(std::string_view text)
    {
        std::vector<Utf8Unit> units;
        units.reserve(text.size());

        std::size_t offset = 0;
        while (offset < text.size())
        {
            Utf8Unit unit = ReadUtf8Unit(text, offset);
            units.push_back(unit);
            offset += unit.length;
        }

        return units;
    }

    static std::vector<std::string> SplitUtf8Strings(std::string_view text)
    {
        std::vector<std::string> output;
        output.reserve(text.size());

        std::size_t offset = 0;
        while (offset < text.size())
        {
            const Utf8Unit unit = ReadUtf8Unit(text, offset);
            output.emplace_back(text.substr(offset, unit.length));
            offset += unit.length;
        }

        return output;
    }

    void BuildByteTables()
    {
        std::vector<int> bytes;
        bytes.reserve(256);

        for (int value = 33; value <= 126; ++value)
        {
            bytes.push_back(value);
        }
        for (int value = 161; value <= 172; ++value)
        {
            bytes.push_back(value);
        }
        for (int value = 174; value <= 255; ++value)
        {
            bytes.push_back(value);
        }

        std::vector<int> codePoints = bytes;
        std::array<bool, 256> included{};
        for (const int value : bytes)
        {
            included[static_cast<std::size_t>(value)] = true;
        }

        int placeholderIndex = 0;
        for (int byteValue = 0; byteValue < 256; ++byteValue)
        {
            if (!included[static_cast<std::size_t>(byteValue)])
            {
                bytes.push_back(byteValue);
                codePoints.push_back(256 + placeholderIndex);
                ++placeholderIndex;
            }
        }

        for (std::size_t index = 0; index < bytes.size(); ++index)
        {
            std::string encodedCharacter;
            AppendUtf8(static_cast<char32_t>(codePoints[index]), encodedCharacter);

            const auto byteValue = static_cast<std::uint8_t>(bytes[index]);
            byteEncoder_[byteValue] = encodedCharacter;
            byteDecoder_[encodedCharacter] = byteValue;
        }
    }

    void LoadVocabulary(const nlohmann::json& model)
    {
        if (!model.contains("vocab") || !model["vocab"].is_object())
        {
            throw std::runtime_error("BPE model does not contain a vocab object");
        }

        TokenId maximumId = -1;
        for (const auto& [token, value] : model["vocab"].items())
        {
            const TokenId tokenId = value.get<TokenId>();
            if (tokenId < 0)
            {
                throw std::runtime_error("Vocabulary contains a negative token ID");
            }

            encoder_[token] = tokenId;
            maximumId = std::max(maximumId, tokenId);
        }

        decoder_.resize(static_cast<std::size_t>(maximumId + 1));
        for (const auto& [token, tokenId] : encoder_)
        {
            auto& slot = decoder_[static_cast<std::size_t>(tokenId)];
            if (slot.has_value())
            {
                throw std::runtime_error("Vocabulary contains duplicate token IDs");
            }
            slot = token;
        }
    }

    void LoadMerges(const nlohmann::json& model)
    {
        if (!model.contains("merges") || !model["merges"].is_array())
        {
            throw std::runtime_error("BPE model does not contain a merges array");
        }

        std::size_t rank = 0;
        for (const auto& merge : model["merges"])
        {
            std::string first;
            std::string second;

            if (merge.is_string())
            {
                const std::string value = merge.get<std::string>();
                const std::size_t separator = value.find(' ');
                if (separator == std::string::npos)
                {
                    throw std::runtime_error("Invalid BPE merge entry: " + value);
                }

                first = value.substr(0, separator);
                second = value.substr(separator + 1);
            }
            else if (merge.is_array() && merge.size() == 2)
            {
                first = merge[0].get<std::string>();
                second = merge[1].get<std::string>();
            }
            else
            {
                throw std::runtime_error("Unsupported BPE merge entry format");
            }

            mergeRanks_.emplace(std::make_pair(std::move(first), std::move(second)), rank);
            ++rank;
        }
    }

    void LoadAddedTokens(const nlohmann::json& root)
    {
        if (!root.contains("added_tokens") || !root["added_tokens"].is_array())
        {
            return;
        }

        for (const auto& token : root["added_tokens"])
        {
            if (!token.contains("content") || !token.contains("id"))
            {
                continue;
            }

            AddedToken added;
            added.content = token["content"].get<std::string>();
            added.id = token["id"].get<TokenId>();
            added.special = token.value("special", false);

            if (added.id < 0)
            {
                throw std::runtime_error("Added token contains a negative token ID");
            }

            if (static_cast<std::size_t>(added.id) >= decoder_.size())
            {
                decoder_.resize(static_cast<std::size_t>(added.id + 1));
            }

            encoder_[added.content] = added.id;
            decoder_[static_cast<std::size_t>(added.id)] = added.content;

            if (added.special)
            {
                specialTokenIds_.insert(added.id);
            }

            addedTokens_.push_back(std::move(added));
        }

        std::sort(
            addedTokens_.begin(),
            addedTokens_.end(),
            [](const AddedToken& left, const AddedToken& right)
            {
                return left.content.size() > right.content.size();
            });
    }

    void EncodeRange(
        std::string_view text,
        std::size_t begin,
        std::size_t end,
        std::vector<TokenId>& output) const
    {
        std::size_t cursor = begin;

        while (cursor < end)
        {
            const AddedToken* bestToken = nullptr;
            std::size_t bestPosition = std::string_view::npos;

            for (const AddedToken& addedToken : addedTokens_)
            {
                const std::size_t position = text.find(addedToken.content, cursor);
                if (position == std::string_view::npos || position >= end)
                {
                    continue;
                }

                if (bestToken == nullptr || position < bestPosition ||
                    (position == bestPosition &&
                     addedToken.content.size() > bestToken->content.size()))
                {
                    bestToken = &addedToken;
                    bestPosition = position;
                }
            }

            const std::size_t plainEnd = bestToken == nullptr ? end : bestPosition;
            if (plainEnd > cursor)
            {
                EncodePlainText(text.substr(cursor, plainEnd - cursor), output);
            }

            if (bestToken == nullptr)
            {
                break;
            }

            output.push_back(bestToken->id);
            cursor = bestPosition + bestToken->content.size();
        }
    }

    static bool IsWhitespace(char32_t codePoint)
    {
        switch (codePoint)
        {
        case U' ':
        case U'\t':
        case U'\n':
        case U'\r':
        case U'\f':
        case U'\v':
        case 0x0085:
        case 0x00A0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return codePoint >= 0x2000 && codePoint <= 0x200A;
        }
    }

    static bool IsNumber(char32_t codePoint)
    {
        if (codePoint >= U'0' && codePoint <= U'9')
        {
            return true;
        }

        static constexpr std::pair<char32_t, char32_t> ranges[] = {
            {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x07C0, 0x07C9},
            {0x0966, 0x096F}, {0x09E6, 0x09EF}, {0x0A66, 0x0A6F},
            {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F}, {0x0BE6, 0x0BEF},
            {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF}, {0x0D66, 0x0D6F},
            {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x0F20, 0x0F29},
            {0x1040, 0x1049}, {0x1090, 0x1099}, {0x17E0, 0x17E9},
            {0x1810, 0x1819}, {0x1946, 0x194F}, {0x19D0, 0x19D9},
            {0x1A80, 0x1A89}, {0x1A90, 0x1A99}, {0x1B50, 0x1B59},
            {0x1BB0, 0x1BB9}, {0x1C40, 0x1C49}, {0x1C50, 0x1C59},
            {0xA620, 0xA629}, {0xA8D0, 0xA8D9}, {0xA900, 0xA909},
            {0xA9D0, 0xA9D9}, {0xA9F0, 0xA9F9}, {0xAA50, 0xAA59},
            {0xABF0, 0xABF9}, {0xFF10, 0xFF19}, {0x104A0, 0x104A9},
            {0x1D7CE, 0x1D7FF}};

        for (const auto& [first, last] : ranges)
        {
            if (codePoint >= first && codePoint <= last)
            {
                return true;
            }
        }

        return false;
    }

    static bool IsSymbolOrPunctuation(char32_t codePoint)
    {
        if (codePoint < 0x80)
        {
            return !std::isalnum(static_cast<unsigned char>(codePoint)) &&
                   !std::isspace(static_cast<unsigned char>(codePoint));
        }

        static constexpr std::pair<char32_t, char32_t> ranges[] = {
            {0x2000, 0x206F}, // General punctuation
            {0x20A0, 0x20CF}, // Currency symbols
            {0x2100, 0x214F}, // Letterlike symbols
            {0x2190, 0x21FF}, // Arrows
            {0x2200, 0x22FF}, // Mathematical operators
            {0x2300, 0x23FF}, // Miscellaneous technical
            {0x2400, 0x243F}, // Control pictures
            {0x2440, 0x245F}, // OCR
            {0x2460, 0x24FF}, // Enclosed alphanumerics
            {0x2500, 0x257F}, // Box drawing
            {0x2580, 0x259F}, // Block elements
            {0x25A0, 0x25FF}, // Geometric shapes
            {0x2600, 0x26FF}, // Miscellaneous symbols
            {0x2700, 0x27BF}, // Dingbats
            {0x27C0, 0x27EF},
            {0x27F0, 0x27FF},
            {0x2900, 0x297F},
            {0x2980, 0x29FF},
            {0x2A00, 0x2AFF},
            {0x2B00, 0x2BFF},
            {0x2E00, 0x2E7F}, // Supplemental punctuation
            {0x3000, 0x303F}, // CJK symbols and punctuation
            {0xFE10, 0xFE1F},
            {0xFE30, 0xFE4F},
            {0xFF01, 0xFF0F},
            {0xFF1A, 0xFF20},
            {0xFF3B, 0xFF40},
            {0xFF5B, 0xFF65},
            {0x1F000, 0x1FAFF}}; // Emoji and pictographs

        for (const auto& [first, last] : ranges)
        {
            if (codePoint >= first && codePoint <= last)
            {
                return true;
            }
        }

        return false;
    }

    static bool IsLetter(char32_t codePoint)
    {
        if ((codePoint >= U'A' && codePoint <= U'Z') ||
            (codePoint >= U'a' && codePoint <= U'z'))
        {
            return true;
        }

        if (codePoint < 0x80 || IsWhitespace(codePoint) || IsNumber(codePoint) ||
            IsSymbolOrPunctuation(codePoint))
        {
            return false;
        }

        // Dependency-free approximation: most remaining non-ASCII code points
        // are treated as letters/marks. This includes accented text, CJK,
        // Hangul, Arabic, Cyrillic, and combining marks.
        return true;
    }

    static bool StartsWithAt(
        std::string_view text,
        std::size_t offset,
        std::string_view value)
    {
        return offset + value.size() <= text.size() &&
               text.substr(offset, value.size()) == value;
    }

    static std::size_t MatchContraction(std::string_view text, std::size_t offset)
    {
        static constexpr std::string_view contractions[] = {
            "'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};

        for (const std::string_view contraction : contractions)
        {
            if (StartsWithAt(text, offset, contraction))
            {
                return contraction.size();
            }
        }

        return 0;
    }

    static std::vector<std::string_view> Pretokenize(std::string_view text)
    {
        std::vector<std::string_view> pieces;
        const std::vector<Utf8Unit> units = SplitUtf8Units(text);

        std::size_t unitIndex = 0;
        while (unitIndex < units.size())
        {
            const std::size_t byteOffset = units[unitIndex].offset;

            if (units[unitIndex].codePoint == U'\'' )
            {
                const std::size_t contractionLength = MatchContraction(text, byteOffset);
                if (contractionLength != 0)
                {
                    pieces.push_back(text.substr(byteOffset, contractionLength));

                    const std::size_t targetOffset = byteOffset + contractionLength;
                    while (unitIndex < units.size() &&
                           units[unitIndex].offset < targetOffset)
                    {
                        ++unitIndex;
                    }
                    continue;
                }
            }

            // GPT-2 regex allows one literal ASCII space before a letter,
            // number, or symbol run.
            bool includeLeadingSpace = false;
            std::size_t groupStartIndex = unitIndex;
            if (units[unitIndex].codePoint == U' ' && unitIndex + 1 < units.size())
            {
                const char32_t next = units[unitIndex + 1].codePoint;
                if (!IsWhitespace(next))
                {
                    includeLeadingSpace = true;
                    groupStartIndex = unitIndex + 1;
                }
            }

            if (includeLeadingSpace)
            {
                const char32_t first = units[groupStartIndex].codePoint;
                std::size_t endIndex = groupStartIndex;

                if (IsLetter(first))
                {
                    while (endIndex < units.size() && IsLetter(units[endIndex].codePoint))
                    {
                        ++endIndex;
                    }
                }
                else if (IsNumber(first))
                {
                    while (endIndex < units.size() && IsNumber(units[endIndex].codePoint))
                    {
                        ++endIndex;
                    }
                }
                else
                {
                    while (endIndex < units.size() &&
                           !IsWhitespace(units[endIndex].codePoint) &&
                           !IsLetter(units[endIndex].codePoint) &&
                           !IsNumber(units[endIndex].codePoint))
                    {
                        ++endIndex;
                    }
                }

                const std::size_t endOffset =
                    endIndex < units.size() ? units[endIndex].offset : text.size();
                pieces.push_back(text.substr(byteOffset, endOffset - byteOffset));
                unitIndex = endIndex;
                continue;
            }

            const char32_t current = units[unitIndex].codePoint;
            std::size_t endIndex = unitIndex;

            if (IsLetter(current))
            {
                while (endIndex < units.size() && IsLetter(units[endIndex].codePoint))
                {
                    ++endIndex;
                }
            }
            else if (IsNumber(current))
            {
                while (endIndex < units.size() && IsNumber(units[endIndex].codePoint))
                {
                    ++endIndex;
                }
            }
            else if (IsWhitespace(current))
            {
                while (endIndex < units.size() && IsWhitespace(units[endIndex].codePoint))
                {
                    ++endIndex;
                }

                // For whitespace followed by non-whitespace, GPT-2's
                // \s+(?!\S) alternative leaves one literal space available
                // to prefix the next piece when possible.
                if (endIndex < units.size() && endIndex - unitIndex > 1 &&
                    units[endIndex - 1].codePoint == U' ')
                {
                    --endIndex;
                }
            }
            else
            {
                while (endIndex < units.size() &&
                       !IsWhitespace(units[endIndex].codePoint) &&
                       !IsLetter(units[endIndex].codePoint) &&
                       !IsNumber(units[endIndex].codePoint))
                {
                    ++endIndex;
                }
            }

            if (endIndex == unitIndex)
            {
                ++endIndex;
            }

            const std::size_t endOffset =
                endIndex < units.size() ? units[endIndex].offset : text.size();
            pieces.push_back(text.substr(byteOffset, endOffset - byteOffset));
            unitIndex = endIndex;
        }

        return pieces;
    }

    void EncodePlainText(
        std::string_view text,
        std::vector<TokenId>& output) const
    {
        for (const std::string_view piece : Pretokenize(text))
        {
            std::string byteMappedPiece;
            byteMappedPiece.reserve(piece.size() * 2);

            for (const unsigned char byte : piece)
            {
                byteMappedPiece.append(byteEncoder_[byte]);
            }

            const std::vector<std::string> bpeTokens = ApplyBpe(byteMappedPiece);
            for (const std::string& bpeToken : bpeTokens)
            {
                const auto found = encoder_.find(bpeToken);
                if (found != encoder_.end())
                {
                    output.push_back(found->second);
                    continue;
                }

                if (unknownTokenId_.has_value())
                {
                    output.push_back(*unknownTokenId_);
                    continue;
                }

                throw std::runtime_error(
                    "BPE produced a token that is absent from the vocabulary: " +
                    bpeToken);
            }
        }
    }

    [[nodiscard]] std::vector<std::string> ApplyBpe(std::string_view token) const
    {
        std::vector<std::string> symbols = SplitUtf8Strings(token);
        if (symbols.size() < 2)
        {
            return symbols;
        }

        while (symbols.size() >= 2)
        {
            std::size_t bestRank = std::numeric_limits<std::size_t>::max();
            std::optional<std::pair<std::string, std::string>> bestPair;

            for (std::size_t index = 0; index + 1 < symbols.size(); ++index)
            {
                const auto candidate = std::make_pair(symbols[index], symbols[index + 1]);
                const auto found = mergeRanks_.find(candidate);
                if (found != mergeRanks_.end() && found->second < bestRank)
                {
                    bestRank = found->second;
                    bestPair = candidate;
                }
            }

            if (!bestPair.has_value())
            {
                break;
            }

            std::vector<std::string> merged;
            merged.reserve(symbols.size());

            std::size_t index = 0;
            while (index < symbols.size())
            {
                if (index + 1 < symbols.size() &&
                    symbols[index] == bestPair->first &&
                    symbols[index + 1] == bestPair->second)
                {
                    merged.push_back(symbols[index] + symbols[index + 1]);
                    index += 2;
                }
                else
                {
                    merged.push_back(std::move(symbols[index]));
                    ++index;
                }
            }

            symbols = std::move(merged);
        }

        return symbols;
    }

    void DecodeByteMappedText(
        std::string_view byteMappedText,
        std::string& output) const
    {
        std::size_t offset = 0;
        while (offset < byteMappedText.size())
        {
            const Utf8Unit unit = ReadUtf8Unit(byteMappedText, offset);
            const std::string encodedCharacter(
                byteMappedText.substr(offset, unit.length));

            const auto found = byteDecoder_.find(encodedCharacter);
            if (found == byteDecoder_.end())
            {
                throw std::runtime_error(
                    "Vocabulary token contains a character outside the GPT-2 byte map");
            }

            output.push_back(static_cast<char>(found->second));
            offset += unit.length;
        }
    }
};

