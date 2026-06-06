import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class BPETokenizer:
    def __init__(self):
        self.vocab = {}  # 最终词表: token -> id
        self.merges = {}  # 合并规则: (token1, token2) -> new_token
        self.token_to_id = {}
        self.id_to_token = {}

    def get_stats(self, corpus: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """统计相邻 pair 出现频率"""
        pairs = defaultdict(int)
        for word in corpus:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def merge_vocab(self, word: List[str], pair: Tuple[str, str]) -> List[str]:
        """合并指定的 pair"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])  # 合并
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def train(self, texts: List[str], vocab_size: int = 10000, min_freq: int = 2):
        """训练 BPE"""
        # 1. 预处理：把文本拆成字符 + 添加结束符
        corpus = []
        for text in texts:
            # 简单预分词（可替换为更复杂的）
            words = re.findall(r"\w+|[^\w\s]", text.lower())
            for word in words:
                # 每个字符作为初始 token
                tokens = list(word) + ["</w>"]
                corpus.append(tokens)

        # 2. 初始化词表
        vocab = Counter()
        for word in corpus:
            vocab[tuple(word)] += 1

        # 当前所有 token
        tokens = set()
        for word in vocab:
            tokens.update(word)

        print("init length: ", len(tokens))
        # 3. 迭代合并
        merges = {}
        while len(tokens) < vocab_size:
            print("=" * 50)
            print("vocab keys: ", vocab.keys())
            pairs = self.get_stats([list(word) for word in vocab.keys()])
            if not pairs:
                break

            # 找到最频繁的 pair
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]

            # 记录合并规则
            new_token = best_pair[0] + best_pair[1]
            merges[best_pair] = new_token
            tokens.add(new_token)

            # 更新 corpus
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = tuple(self.merge_vocab(list(word), best_pair))
                new_vocab[new_word] = freq
            vocab = new_vocab

        # 构建最终词表
        self.merges = merges
        sorted_tokens = sorted(list(tokens))
        self.token_to_id = {token: i for i, token in enumerate(sorted_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(sorted_tokens)}
        self.vocab = self.token_to_id

        print(self.vocab)

        print(f"BPE 训练完成！最终词表大小: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """文本编码为 token ids"""
        # 简单分词
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        token_ids = []

        for word in words:
            tokens = list(word) + ["</w>"]

            # 按合并规则逐步合并
            while len(tokens) > 1:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                # 找到可合并的 pair 中优先级最高的
                can_merge = [
                    (pair, self.merges.get(pair))
                    for pair in pairs
                    if pair in self.merges
                ]
                if not can_merge:
                    break
                # 合并优先级最高的（最早学习的）
                best_pair = min(
                    can_merge, key=lambda x: list(self.merges.keys()).index(x[0])
                )[0]
                tokens = self.merge_vocab(tokens, best_pair)

            # 转 id
            for t in tokens:
                token_ids.append(
                    self.token_to_id.get(t, self.token_to_id.get("<unk>", 0))
                )

        return token_ids

    def decode(self, ids: List[int]) -> str:
        """token ids 解码为文本"""
        tokens = [self.id_to_token[i] for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()


# 示例数据
texts = [
    "hello world",
    "hello python",
    "python is awesome",
    "machine learning is fascinating",
    "I love natural language processing",
] * 10  # 重复增加数据量

# 训练
tokenizer = BPETokenizer()
tokenizer.train(texts, vocab_size=500, min_freq=1)

# 测试
text = "hello python machine learning"
ids = tokenizer.encode(text)
print("Token IDs:", ids)
print("Decoded:", tokenizer.decode(ids))
