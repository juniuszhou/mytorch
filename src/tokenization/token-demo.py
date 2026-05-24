import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "Hello, world!"
tokens = tokenizer.tokenize(text)  # WordPiece 分词
print(tokens)  # 输出: ['hello', ',', 'world', '!']

# 转换为 ID
inputs = tokenizer(text, return_tensors="pt")
print(inputs["input_ids"])  # tensor([[ 101, 7592, 1010, 2088,  999,  102]])


text = "Hello, world! How are you?"
inputs = tokenizer(text, return_tensors="pt")
print(inputs["input_ids"])


class CustomTokenizer:
    def __init__(self, vocab_file=None, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )  # 或加载自定义 vocab
        self.max_length = max_length

    def tokenize(self, text):
        # 预分词 + 子词拆分
        tokens = self.tokenizer.tokenize(text)
        # 添加特殊 token
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # ID 映射
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # 填充/截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]  # 截断
        else:
            input_ids += [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            )  # 填充
        # 掩码
        attention_mask = [
            1 if i != self.tokenizer.pad_token_id else 0 for i in input_ids
        ]
        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

    def detokenize(self, tokens):
        # 逆向：ID → token → 文本
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


# 使用示例
tokenizer = CustomTokenizer()
result = tokenizer.tokenize("Hello, world! This is a test.")
print(result["input_ids"].shape)  # torch.Size([1, 512])
print(result["input_ids"])
print(
    tokenizer.detokenize(result["input_ids"][0].tolist())
)  # "Hello, world! This is a test."
