import torch
import torch.nn as nn

class KVCache:
    def __init__(self, max_length: int, dtype=torch.float16, device="cuda"):
        self.max_length = max_length
        self.key_cache = None
        self.value_cache = None
        self.dtype = dtype
        self.device = device
        self.current_length = 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """更新 KV Cache"""
        bsz, num_heads, seq_len, head_dim = k.shape
        
        if self.key_cache is None:
            # 第一次初始化
            self.key_cache = torch.zeros(
                (bsz, num_heads, self.max_length, head_dim),
                dtype=self.dtype, device=self.device
            )
            self.value_cache = torch.zeros(
                (bsz, num_heads, self.max_length, head_dim),
                dtype=self.dtype, device=self.device
            )
        
        # 把新的 k, v 写入 cache
        self.key_cache[:, :, self.current_length:self.current_length+seq_len] = k
        self.value_cache[:, :, self.current_length:self.current_length+seq_len] = v
        
        self.current_length += seq_len
        return self.key_cache[:, :, :self.current_length], self.value_cache[:, :, :self.current_length]

    def clear(self):
        self.current_length = 0



import torch
import torch.nn as nn

class KVCache:
    def __init__(self, max_length: int, dtype=torch.float16, device="cuda"):
        self.max_length = max_length
        self.key_cache = None
        self.value_cache = None
        self.dtype = dtype
        self.device = device
        self.current_length = 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """更新 KV Cache"""
        bsz, num_heads, seq_len, head_dim = k.shape
        
        if self.key_cache is None:
            # 第一次初始化
            self.key_cache = torch.zeros(
                (bsz, num_heads, self.max_length, head_dim),
                dtype=self.dtype, device=self.device
            )
            self.value_cache = torch.zeros(
                (bsz, num_heads, self.max_length, head_dim),
                dtype=self.dtype, device=self.device
            )
        
        # 把新的 k, v 写入 cache
        self.key_cache[:, :, self.current_length:self.current_length+seq_len] = k
        self.value_cache[:, :, self.current_length:self.current_length+seq_len] = v
        
        self.current_length += seq_len
        return self.key_cache[:, :, :self.current_length], self.value_cache[:, :, :self.current_length]

    def clear(self):
        self.current_length = 0


import torch
import torch.nn as nn

class KVCache:
    def __init__(self, max_length: int, dtype=torch.float16, device="cuda"):
        self.max_length = max_length
        self.key_cache = None
        self.value_cache = None
        self.dtype = dtype
        self.device = device
        self.current_length = 0

    def update(self, k: torch.Tensor, v: torch.Tensor):
        """更新 KV Cache"""
        bsz, num_heads, seq_len, head_dim = k.shape
        
        if self.key_cache is None:
            # 第一次初始化
            self.key_cache = torch.zeros(
                (bsz, num_heads, self.max_length, head_dim),
                dtype=self.dtype, device=self.device
            )
            self.value_cache = torch.zeros(
                (bsz, num_heads, self.max_length, head_dim),
                dtype=self.dtype, device=self.device
            )
        
        # 把新的 k, v 写入 cache
        self.key_cache[:, :, self.current_length:self.current_length+seq_len] = k
        self.value_cache[:, :, self.current_length:self.current_length+seq_len] = v
        
        self.current_length += seq_len
        return self.key_cache[:, :, :self.current_length], self.value_cache[:, :, :self.current_length]

    def clear(self):
        self.current_length = 0




