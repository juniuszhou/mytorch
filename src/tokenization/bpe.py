# byte pair encoding

from typing import Dict, List, Tuple


def tokenize(ids):
    pairs = zip(ids[:-1], ids[1:])

    pairs: List[Tuple[int, int]] = [(pair[0], pair[1]) for pair in pairs]

    pair_freq: Dict[Tuple[int, int], int] = {}
    max_freq = 0
    for pair in pairs:
        if pair_freq.get(pair) is None:
            pair_freq[pair] = 0
        pair_freq[pair] += 1
        if pair_freq[pair] > max_freq:
            max_freq = pair_freq[pair]
    result: List[Tuple[int, int]] = []
    for pair in pair_freq:
        if pair_freq[pair] == max_freq:
            result.append(pair)
    return result


def query(item_to_id, id_to_item, ids):
    print("query", ids)

    result = ()
    if isinstance(ids, tuple):
        for id in ids:
            result += query(item_to_id, id_to_item, id)
        return result
    else:
        if ids >= 256:
            return query(item_to_id, id_to_item, id_to_item[ids])
        else:
            return (chr(ids),)


def flatten_tuple(t):
    """将嵌套元组展平，返回一个扁平的 tuple"""
    result = []
    for item in t:
        if isinstance(item, tuple):
            result.extend(flatten_tuple(item))  # 递归
        else:
            result.append(item)
    return tuple(result)


def detokenize(tokens):
    return " ".join(tokens)


def train_bpe(text):
    tokens = tokenize(text)
    return tokens


if __name__ == "__main__":
    text = "low, lower, lowest, very low."

    ids: List[int] = [ord(item) for item in text]
    print(ids)

    new_ids = ids

    # key is tuple of id, value is index
    item_to_id: Dict[str, int] = {}
    id_to_item: Dict[int, str] = {}

    for i in range(256):
        item_to_id[chr(i)] = i
        id_to_item[i] = chr(i)
    a = 0

    while a < 5:
        tokens = tokenize(new_ids)
        print("tokens: ", tokens)
        for token in tokens:
            new_token = ""
            print("token: ", token)
            for t in token:
                new_token += id_to_item[t]
            print(new_token)
            if item_to_id.get(new_token) is None:
                item_to_id[new_token] = len(item_to_id)
                id_to_item[len(id_to_item)] = new_token

        print("item_to_id: ", item_to_id)
        print("id_to_item: ", id_to_item)

        tmp_ids = new_ids.copy()
        new_ids = []
        # skip next one if current one is merged
        skip = False
        for index in range(len(tmp_ids) - 1):
            if skip:
                skip = False
                continue

            pair = id_to_item[tmp_ids[index]] + id_to_item[tmp_ids[index + 1]]
            print("new pair: ", pair)
            if item_to_id.get(pair) is None:
                new_ids.append(tmp_ids[index])
            else:
                print("else here ")
                new_ids.append(item_to_id[pair])
                skip = True
        new_ids.append(tmp_ids[-1])
        print(new_ids)

        for id in new_ids:
            print(id_to_item[id], end="")

        a += 1
        # # print("-" * 50)
        # print("\n")

    # print(flatten_tuple(query(item_to_id, id_to_item, (260,))))
    print()
