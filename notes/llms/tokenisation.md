# Tokenisation

Tokenisation breaks text up into smaller pieces called tokens. Each token is assigned a number and those encoded numbers are what's used as inputs to LLMs via a fixed vocabulary lookup table. The vocabulary size is the total number of unique tokens a model can process. Larger vocab size means more fine-grained data, but more tokens which results in larger compute requirements.

There are various ways to tokenise text, per word, per character, etc.



## Byte-Pair Encoding (BPE)

The most commonly used tokenisation scheme is BPE. This determines tokens based on frequency of n-grams of increasing length.

### Training steps

1. break down training corpus into individual UTF-8 bytes (think of them like characters)
2. count the co-occurrence frequencies of all adjacent token pairs across the entire corpus
3. the most frequent pairs get merged into a new token (e.g. "th"), and added to the vocab
4. re-count the co-occurrence (after N merges) using the new token (e.g. `th` is now added on top of our UTF-8 byte vocab)
5. continue merging until a pre-defined number of merge operations occur, resulting in a fixed length target vocabulary size

Once a BPE tokenizer is trained, the inference stage need to convert text into tokens.

A basic approach can use a greedy longest-match-first approach to convert text into tokens (giving preference to longer tokens). Essentially, given some text, start with the first character, then keep appending characters as long as the character sequence is in the vocabulary.

The primary approach is to split the input text into base UTF-8 bytes and iteratively apply the learned merge rules in their strict chronological priority order (highest-ranked/earliest merges first) across adjacent pairs until no more valid merges can be made.
