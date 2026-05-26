[List for learning LLM](https://github.com/)

- basic process from taiwan university
- langchain for application
- course from karpathy
- course from deeplearning.ai
- course from stanford university
- course from LLM course github https://github.com/mlabonne/llm-course

Dene before chinese new year.

# course in deeplearningai

## post training

https://learn.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training/lesson/rm98gb4q/conversation-between-sharon-zhou-and-andrew-ng

## pre training.

https://learn.deeplearning.ai/courses/pretraining-llms/lesson/z2ntd/introduction

## transformer

https://learn.deeplearning.ai/courses/how-transformer-llms-work/lesson/nfshb/introduction

https://jalammar.github.io/illustrated-transformer/

# course online

## https://github.com/mlabonne/llm-course

# code

## nanoGPT most simple transformer implementation

https://github.com/karpathy/nanoGPT

https://www.youtube.com/watch?v=kCc8FmEb1nY

https://machinelearningmastery.com/the-roadmap-for-mastering-language-models-in-2025/

## minimind

https://github.com/jingyaogong/minimind

https://github.com/jingyaogong/minimind-v

## tokenization exercise

https://github.com/karpathy/minbpe/blob/master/exercise.md

# youtube

https://www.youtube.com/watch?v=bOlVx5zeHLM

https://www.youtube.com/watch?v=VlA_jt_3Qc4 from standford university

### complete the course from taiwan university at first.

https://www.youtube.com/watch?v=VuQUF1VVX40&list=PLJV_el3uVTsMMGi5kbnKP5DrDHZpTX0jT&index=1 (current done before 2025-12-07)

prompt, vocab, token. chat template, context engineering.
user prompt, system prompt, assistant prompt.

the source of LLMs 语言学习资料

1. online data 网上数据
2. label data 标注数据
3. user feedback 用户反馈

application will change the prompt from user. chat template
for example, you input "what is your name?", the application will change the prompt to
"user ask: what is your name?" AI answer: "

some context could be added into the prompt, like the date, time, location, model name, etc.

problem for generate voice or photo or video

1. too many data, much larger than text data. so at first, we need encode the data into token.
2. so encoder ussally encode a 16 _ 16 pixel into a token. then decoder will decode the token into a 16 _ 16 pixel.
