# course in deeplearningai

## transformer illuatration with digram
https://jalammar.github.io/illustrated-transformer/

# course online
## https://github.com/mlabonne/llm-course

# code
## nanoGPT most simple transformer implementation
https://github.com/karpathy/nanoGPT


## tokenization exercise
https://github.com/karpathy/minbpe/blob/master/exercise.md

# standford
transfromer from scratch
https://cs336.stanford.edu/

diffusion model
https://cs296.stanford.edu/

transfromer
https://cs295.stanford.edu/

## minimind a complete model training process.

https://github.com/jingyaogong/minimind

https://github.com/jingyaogong/minimind-v





## questions
1. 在计算attention的时候，为什么不把input的d_model 按照head来split，然后计算。
原因是GPU在计算连续内存的效率更高。
现在流行的方法是把qkv放到一个Linear去计算

