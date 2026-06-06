"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import sys
from pathlib import Path

# This file is named regex.py, so avoid shadowing the third-party `regex` package.
_module_dir = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_module_dir):
    sys.path.pop(0)

import regex as re

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def main():
    input_text = "Hello, world!"
    tokens = re.compile(GPT2_SPLIT_PATTERN).findall(input_text)
    print(tokens)


if __name__ == "__main__":
    main()
