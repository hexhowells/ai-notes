# Tokenisation

Tokenisation breaks text up into smaller pieces called tokens. Each token is assigned a number and those encoded numbers are what's used as inputs to LLMs via a fixed vocabulary lookup table. The vocabulary size is the total number of unique tokens a model can process. Larger vocab size means more fine-grained data, but more tokens which results in larger compute requirements.

There are various ways to tokenise text, per word, per character, etc.



## Byte-Pair Encoding (BPE)

The most commonly used tokenisation scheme is byte-pair encoding (BPE). This determines tokens based on frequency of n-grams of increasing length.

### Training steps

1. break down training corpus into individual UTF-8 bytes (think of them like characters)
2. count the co-occurrence frequencies of all adjacent token pairs across the entire corpus
3. the most frequent pairs get merged into a new token (e.g. "th"), and added to the vocab
4. re-count the co-occurrence (after N merges) using the new token (e.g. `th` is now added on top of our UTF-8 byte vocab)
5. continue merging until a pre-defined number of merge operations occur, resulting in a fixed length target vocabulary size

Once a BPE tokenizer is trained, the inference stage need to convert text into tokens.

A basic approach can use a greedy longest-match-first approach to convert text into tokens (giving preference to longer tokens). Essentially, given some text, start with the first character, then keep appending characters as long as the character sequence is in the vocabulary.

The primary approach is to split the input text into base UTF-8 bytes and iteratively apply the learned merge rules in their strict chronological priority order (highest-ranked/earliest merges first) across adjacent pairs until no more valid merges can be made.

## Special Tokens

On top of the computed tokens from a training dataset, various special tokens can also be added to the vocabulary to trigger certain events around the LLM wrapper and training. These include things like: 

- `<pad>` used to pad sequences to uniform length
- `<|endoftext|>` signals the model to stop generating text
- `<|im_start|>`, `<|im_end|>`, `[INST]` that delineate user prompts, system instructions, etc.

## Glitch Tokens

Early in LLM development, the training data for BPE and pre-training were different. Often with pre-training datasets being cleaner. As a result, the BPE training could generate unique tokens that were common in the messy tokenisation training set, but not present at all in the pre-training set. As a result, using these tokens with the trained LLM model would result in weird outputs as the tokens were not near anything else in feature space, so the LLM wouldn't know what to output.

## Byte Latent Transformer (BLT)

The paper [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) - meta introduced a new tokenisation strategy that dynamically created tokens based on the context of the text. The method used a small byte-level language model across the raw byte stream, with each byte the model calculates predictive entropy (it's uncertainty about the next token), when entropy is high (text is unpredictable) a patch boundary is triggered and a new token is formed. This method groups tokens that are highly predictable (common words, etc) and splits text into smaller tokens where information is less predictable, thus likely containing more useful information.

## Links

[OpenAI online tokenizer](https://platform.openai.com/tokenizer)

[Token speed visualiser](https://mikeveerman.github.io/tokenspeed/?rate=30&mode=code)

