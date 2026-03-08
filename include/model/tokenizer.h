#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <climits>
#include "third_party/json.hpp"

struct Tokenizer {
public:
    // Special token IDs (Qwen3)
    int64_t bos_id = 151643;   // <|endoftext|>
    int64_t eos_id = 151645;   // <|im_end|>
    int64_t im_start_id = 151644;
    int64_t im_end_id   = 151645;

    bool load(const std::string& tokenizer_json_path) {
        std::ifstream f(tokenizer_json_path);
        if (!f.is_open()) {
            fprintf(stderr, "[Tokenizer] Cannot open %s\n", tokenizer_json_path.c_str());
            return false;
        }
        nlohmann::json tj = nlohmann::json::parse(f);

        // Build byte-to-unicode table (GPT-2 bytes_to_unicode())
        build_byte_to_char();

        // Build reverse: unicode char (UTF-8) → original byte
        for (int b = 0; b < 256; b++)
            char_to_byte_[byte_char_[b]] = static_cast<uint8_t>(b);

        // Vocab: token string → id
        int64_t max_id = 0;
        for (auto& [tok, id] : tj["model"]["vocab"].items())
            max_id = std::max(max_id, id.get<int64_t>());
        for (auto& at : tj["added_tokens"])
            max_id = std::max(max_id, at["id"].get<int64_t>());

        id_to_token_.resize(max_id + 1);
        is_special_.resize(max_id + 1, false);
        vocab_.reserve(max_id + 1);

        for (auto& [tok, id] : tj["model"]["vocab"].items()) {
            int64_t i = id.get<int64_t>();
            vocab_[tok] = i;
            id_to_token_[i] = tok;
        }
        for (auto& at : tj["added_tokens"]) {
            int64_t i = at["id"].get<int64_t>();
            std::string tok = at["content"].get<std::string>();
            vocab_[tok] = i;
            id_to_token_[i] = tok;
            is_special_[i]   = true;
            special_tokens_.push_back({tok, i});
        }
        // Sort longest-first so greedy matching prefers longer special tokens
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const SpecTok& a, const SpecTok& b){ return a.str.size() > b.str.size(); });

        // Merge ranks: (left + '\0' + right) → rank  (O(1) lookup)
        auto& jmerges = tj["model"]["merges"];
        merge_rank_.reserve(jmerges.size());
        for (int r = 0; r < (int)jmerges.size(); r++) {
            std::string left  = jmerges[r][0].get<std::string>();
            std::string right = jmerges[r][1].get<std::string>();
            merge_rank_[left + '\0' + right] = r;
        }

        return true;
    }

    // Encode text → token IDs.
    // Special tokens (e.g. <|im_start|>) are matched greedily as atomic units
    // before BPE is applied to the remaining text segments.
    std::vector<int64_t> encode(const std::string& text) const {
        std::vector<int64_t> ids;
        int i = 0, n = text.size();

        while (i < n) {
            // Try to match a special token at current position (longest-first)
            bool hit = false;
            for (auto& st : special_tokens_) {
                int slen = st.str.size();
                if (i + slen <= n && text.compare(i, slen, st.str) == 0) {
                    ids.push_back(st.id);
                    i += slen;
                    hit = true;
                    break;
                }
            }
            if (hit) continue;

            // Find the end of this normal-text segment (stop before next special token)
            int j = i + 1;
            while (j < n) {
                bool at_special = false;
                for (auto& st : special_tokens_) {
                    int slen = st.str.size();
                    if (j + slen <= n && text.compare(j, slen, st.str) == 0) {
                        at_special = true; break;
                    }
                }
                if (at_special) break;
                j++;
            }

            // BPE-encode the segment [i, j)
            for (auto& pre : pretokenize(text.substr(i, j - i))) {
                std::string enc = bytes_to_chars(pre);
                for (auto& piece : bpe_encode(enc)) {
                    auto it = vocab_.find(piece);
                    if (it != vocab_.end())
                        ids.push_back(it->second);
                    else
                        fprintf(stderr, "[Tokenizer] Unknown piece: '%s'\n", piece.c_str());
                }
            }
            i = j;
        }
        return ids;
    }

    // Add a special token by its string content (returns its ID)
    int64_t encode_special(const std::string& tok) const {
        auto it = vocab_.find(tok);
        return (it != vocab_.end()) ? it->second : -1;
    }

    // Decode single token ID → UTF-8 string (empty string for special tokens)
    std::string decode_token(int64_t id) const {
        if (id < 0 || id >= (int64_t)id_to_token_.size()) return "";
        if (is_special_[id]) return "";           // skip special tokens in output
        return chars_to_bytes(id_to_token_[id]);
    }

    // Decode list of token IDs → UTF-8 string
    std::string decode(const std::vector<int64_t>& ids) const {
        std::string out;
        for (int64_t id : ids) out += decode_token(id);
        return out;
    }

    int vocab_size() const { return (int)id_to_token_.size(); }

    bool is_special(int64_t id) const {
        return id >= 0 && id < (int64_t)is_special_.size() && is_special_[id];
    }

    // Build a Qwen3-chat prompt for a single user message.
    // Includes the standard system prompt.  enable_thinking=true adds the
    // "<think>\n\n</think>\n" prefix that coaxes the model into non-thinking mode.
    std::string chat_prompt(const std::string& user_message,
                            bool enable_thinking = false) const {
        std::string s;
        s += "<|im_start|>system\n";
        s += "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        s += "<|im_end|>\n";
        s += "<|im_start|>user\n";
        s += user_message;
        s += "<|im_end|>\n";
        s += "<|im_start|>assistant\n";
        if (!enable_thinking)
            s += "<think>\n\n</think>\n";   // empty think block → skip CoT
        return s;
    }

private:
    struct SpecTok { std::string str; int64_t id; };

    std::unordered_map<std::string, int64_t>  vocab_;
    std::vector<std::string>                  id_to_token_;
    std::vector<bool>                         is_special_;
    std::vector<SpecTok>                      special_tokens_;  // sorted longest-first
    std::unordered_map<std::string, int>      merge_rank_;      // "left\0right" → rank
    std::string                               byte_char_[256];
    std::unordered_map<std::string, uint8_t>  char_to_byte_;

    // ── Byte-to-unicode (GPT-2 bytes_to_unicode) ────────────────────────────

    static std::string cp_to_utf8(uint32_t cp) {
        std::string s;
        if      (cp < 0x80)    { s.push_back(cp); }
        else if (cp < 0x800)   { s.push_back(0xC0|(cp>>6)); s.push_back(0x80|(cp&0x3F)); }
        else                   { s.push_back(0xE0|(cp>>12));
                                 s.push_back(0x80|((cp>>6)&0x3F));
                                 s.push_back(0x80|(cp&0x3F)); }
        return s;
    }

    void build_byte_to_char() {
        // Bytes that map to themselves (printable Latin-1 subset)
        bool in_bs[256] = {};
        for (int c = '!'; c <= '~'; c++) in_bs[c] = true;
        for (int c = 0xA1; c <= 0xAC; c++) in_bs[c] = true;
        for (int c = 0xAE; c <= 0xFF; c++) in_bs[c] = true;
        // Remaining 68 bytes map to codepoints 256..323
        int n = 0;
        for (int b = 0; b < 256; b++) {
            byte_char_[b] = in_bs[b] ? cp_to_utf8(b) : cp_to_utf8(256 + n++);
        }
    }

    // Encode each byte of s using byte_char_ mapping
    std::string bytes_to_chars(const std::string& s) const {
        std::string r;
        for (unsigned char c : s) r += byte_char_[c];
        return r;
    }

    // Decode byte-level unicode string back to UTF-8
    // Iterates UTF-8 chars and maps each back via char_to_byte_
    std::string chars_to_bytes(const std::string& s) const {
        std::string r;
        int i = 0, n = s.size();
        while (i < n) {
            unsigned char c = s[i];
            int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            std::string ch = s.substr(i, len);
            auto it = char_to_byte_.find(ch);
            if (it != char_to_byte_.end()) r.push_back(it->second);
            else r += ch;   // pass-through for chars not in the table
            i += len;
        }
        return r;
    }

    // ── Pre-tokenizer ────────────────────────────────────────────────────────
    // Approximates GPT-2 regex for ASCII text:
    //   contractions | [non-word]?letters | single-digit |
    //   [space]?punctuation | whitespace
    //
    // Space is attached to the BEGINNING of the following word (Ġ convention).

    static bool is_letter(char c)  { return std::isalpha((unsigned char)c); }
    static bool is_digit(char c)   { return std::isdigit((unsigned char)c); }
    static bool is_space(char c)   { return c == ' ' || c == '\t'; }
    static bool is_newline(char c) { return c == '\r' || c == '\n'; }

    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        int i = 0, n = text.size();

        while (i < n) {
            int start = i;

            // --- Contractions: 's 't 're 've 'm 'll 'd ---
            if (text[i] == '\'') {
                if (i+1 < n) {
                    char nc = tolower((unsigned char)text[i+1]);
                    if (nc=='s' || nc=='t' || nc=='m' || nc=='d') {
                        tokens.push_back(text.substr(i, 2)); i+=2; continue;
                    }
                }
                if (i+2 < n) {
                    char a = tolower((unsigned char)text[i+1]);
                    char b = tolower((unsigned char)text[i+2]);
                    if ((a=='r'&&b=='e') || (a=='v'&&b=='e') || (a=='l'&&b=='l')) {
                        tokens.push_back(text.substr(i, 3)); i+=3; continue;
                    }
                }
            }

            // --- Newlines: [\r\n]+ ---
            if (is_newline(text[i])) {
                while (i < n && is_newline(text[i])) i++;
                tokens.push_back(text.substr(start, i-start)); continue;
            }

            // --- [optional non-word char] + letters ---
            // The optional char is a single non-letter, non-digit, non-newline (e.g. a space)
            {
                int j = i;
                if (!is_letter(text[j]) && !is_digit(text[j]) && !is_newline(text[j]) &&
                    (j+1 < n) && is_letter(text[j+1]))
                    j++;  // consume optional prefix (e.g. space before word)

                if (j < n && is_letter(text[j])) {
                    while (j < n && is_letter(text[j])) j++;
                    tokens.push_back(text.substr(i, j-i)); i=j; continue;
                }
                // If we consumed a prefix but found no letters, fall through with i unchanged
            }

            // --- Single digit ---
            if (is_digit(text[i])) {
                tokens.push_back(text.substr(i, 1)); i++; continue;
            }

            // --- [optional space] + punctuation/symbols ---
            {
                int j = i;
                if (j < n && text[j]==' ' && j+1 < n &&
                    !is_space(text[j+1]) && !is_letter(text[j+1]) && !is_digit(text[j+1]))
                    j++;  // consume optional leading space
                if (j < n && !is_space(text[j]) && !is_letter(text[j]) && !is_digit(text[j]) &&
                    !is_newline(text[j])) {
                    while (j < n && !is_space(text[j]) && !is_letter(text[j]) &&
                           !is_digit(text[j]) && !is_newline(text[j])) j++;
                    while (j < n && is_newline(text[j])) j++;  // trailing newlines
                    tokens.push_back(text.substr(i, j-i)); i=j; continue;
                }
            }

            // --- Trailing/lone spaces ---
            if (is_space(text[i])) {
                while (i < n && is_space(text[i])) i++;
                tokens.push_back(text.substr(start, i-start)); continue;
            }

            // Fallback: single character
            tokens.push_back(text.substr(i, 1)); i++;
        }
        return tokens;
    }

    // ── BPE encode ───────────────────────────────────────────────────────────

    // Split UTF-8 string into individual characters (each as a std::string)
    static std::vector<std::string> split_utf8(const std::string& s) {
        std::vector<std::string> chars;
        int i = 0, n = s.size();
        while (i < n) {
            unsigned char c = s[i];
            int len = (c < 0x80)?1:(c < 0xE0)?2:(c < 0xF0)?3:4;
            chars.push_back(s.substr(i, len));
            i += len;
        }
        return chars;
    }

    std::vector<std::string> bpe_encode(const std::string& pre_token) const {
        auto word = split_utf8(pre_token);
        if (word.size() <= 1) return word;

        // Greedy BPE: each pass finds the globally lowest-rank merge and applies
        // all occurrences of it in left-to-right order.
        while (true) {
            int best_rank = INT_MAX, best_i = -1;
            for (int j = 0; j+1 < (int)word.size(); j++) {
                auto it = merge_rank_.find(word[j] + '\0' + word[j+1]);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i    = j;
                }
            }
            if (best_i < 0) break;

            // Apply all occurrences of (word[best_i], word[best_i+1]) left-to-right
            const std::string left  = word[best_i];
            const std::string right = word[best_i+1];
            std::string merged = left + right;

            std::vector<std::string> nw;
            for (int j = 0; j < (int)word.size(); ) {
                if (j+1 < (int)word.size() && word[j]==left && word[j+1]==right) {
                    nw.push_back(merged); j += 2;
                } else {
                    nw.push_back(word[j]); j++;
                }
            }
            word = std::move(nw);
        }
        return word;
    }
};
