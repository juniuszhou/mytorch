# tokenization

## issue list for it

### https://www.youtube.com/watch?v=zduSFxRajkE&t=476s course video

intro: Tokenization, GPT-2 paper, tokenization-related issues
tokenization by example in a Web UI (tiktokenizer)
strings in Python, Unicode code points
Unicode byte encodings, ASCII, UTF-8, UTF-16, UTF-32
daydreaming: deleting tokenization
Byte Pair Encoding (BPE) algorithm walkthrough
starting the implementation
counting consecutive pairs, finding most common pair
merging the most common pair
training the tokenizer: adding the while loop, compression ratio
tokenizer/LLM diagram: it is a completely separate stage
decoding tokens to strings
encoding strings to tokens
regex patterns to force splits across categories
tiktoken library intro, differences between GPT-2/GPT-4 regex
GPT-2 encoder.py released by OpenAI walkthrough
special tokens, tiktoken handling of, GPT-2/GPT-4 differences
minbpe exercise time! write your own GPT-4 tokenizer
sentencepiece library intro, used to train Llama 2 vocabulary
how to set vocabulary set? revisiting gpt.py transformer
training new tokens, example of prompt compression
multimodal [image, video, audio] tokenization with vector quantization
revisiting and explaining the quirks of LLM tokenization
final recommendations