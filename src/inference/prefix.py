"""
Prefix KV caching for LLM-style inference (educational demo).

Many requests share the same long system prompt or document prefix. You can
run the transformer once on that prefix, store per-layer (K, V) tensors, and
reuse them on later calls so you only compute K/V for new tokens.

This file uses a tiny toy model where keys/values are linear projections of
token embeddings (not a full decoder stack). The important part is the same as
in production: hash(prefix_token_ids) -> lookup or compute -> reuse cached K/V.
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn

# One layer's KV: (key, value), each (batch, num_heads, seq_len, head_dim)
KVPair = Tuple[torch.Tensor, torch.Tensor]
# Full model state: list of KVPair, one per layer
KVCache = list[KVPair]


def hash_prefix(prefix_tokens: torch.Tensor) -> int:
    """Stable hash for a prefix token tensor (shape + ids), no NumPy dependency."""
    t = prefix_tokens.detach().cpu().to(torch.int64).contiguous()
    return hash((tuple(t.shape), tuple(t.flatten().tolist())))


def clone_kv_cache(kv: KVCache) -> KVCache:
    return [(k.clone(), v.clone()) for k, v in kv]


class PrefixKVCache:
    """Maps prefix token ids -> frozen K/V tensors (per layer) for that prefix."""

    def __init__(self, *, verbose: bool = False) -> None:
        # key: prefix_hash, value: KVCache (cloned tensors, safe to reuse)
        self.prefix_cache: dict[int, KVCache] = {}
        self.verbose = verbose

    def compute_or_get_prefix(
        self, prefix_tokens: torch.Tensor, model: nn.Module
    ) -> KVCache:
        prefix_hash = hash_prefix(prefix_tokens)

        if prefix_hash not in self.prefix_cache:
            with torch.no_grad():
                _, kv_cache = model.forward_prefix(prefix_tokens)
            self.prefix_cache[prefix_hash] = clone_kv_cache(kv_cache)
            if self.verbose:
                print("Prefix cache miss — computed and stored.")
        elif self.verbose:
            print("Prefix cache hit — reused stored K/V.")

        return self.prefix_cache[prefix_hash]

    def forward_with_prefix(
        self,
        model: nn.Module,
        full_input_ids: torch.Tensor,
        prefix_len: int,
    ) -> KVCache:
        """
        Return full-sequence K/V per layer: concat(prefix_kv, suffix_kv) on seq dim.

        `model.forward_suffix` must return K/V only for suffix positions, using the
        same projection as the prefix pass (toy model); a real LLM would run a
        proper forward with past_key_values instead.
        """
        prefix_tokens = full_input_ids[:, :prefix_len]
        suffix_tokens = full_input_ids[:, prefix_len:]
        prefix_kv = self.compute_or_get_prefix(prefix_tokens, model)

        with torch.no_grad():
            suffix_kv = model.forward_suffix(suffix_tokens, prefix_kv)

        combined: KVCache = []
        for (pk, pv), (sk, sv) in zip(prefix_kv, suffix_kv, strict=True):
            combined_k = torch.cat([pk, sk], dim=2)
            combined_v = torch.cat([pv, sv], dim=2)
            combined.append((combined_k, combined_v))
        return combined


class TinyPrefixModel(nn.Module):
    """
    Minimal multi-layer "encoder-style" stack: each position's K/V depend only on
    its embedding. That is not true for real decoder layers, but it matches the
    concat picture and keeps the demo small.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        num_layers: int = 4,
        d_model: int = 64,
        num_heads: int = 4,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.device = torch.device(device)
        self.dtype = dtype

        self.embed = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.proj_kv = nn.ModuleList(
            nn.Linear(d_model, 2 * d_model, device=device, dtype=dtype)
            for _ in range(num_layers)
        )

    def _tokens_to_kv(self, x: torch.Tensor) -> KVCache:
        """x: (batch, seq, d_model) -> list of (k, v) per layer."""
        bsz, seq_len, _ = x.shape
        out: KVCache = []
        for lin in self.proj_kv:
            kv = lin(x)  # (B, L, 2 * D)
            k, v = kv.chunk(2, dim=-1)
            k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            out.append((k.contiguous(), v.contiguous()))
        return out

    def forward_prefix(self, prefix_ids: torch.Tensor):
        """Run only prefix positions; return logits unused and per-layer KV cache."""
        x = self.embed(prefix_ids)
        kv_cache = self._tokens_to_kv(x)
        return None, kv_cache

    def forward_suffix(self, suffix_ids: torch.Tensor, prefix_kv: KVCache) -> KVCache:
        """
        Compute K/V for suffix token positions only. `prefix_kv` is ignored by this toy
        (real models need it for attention); it is kept in the signature to mirror
        production APIs where the decoder consumes past_key_values.
        """
        _ = prefix_kv  # Real decoder attention would use cached prefix keys/values here.
        if suffix_ids.shape[1] == 0:
            bsz = suffix_ids.shape[0]
            return [
                (
                    torch.zeros(
                        bsz,
                        self.num_heads,
                        0,
                        self.head_dim,
                        device=suffix_ids.device,
                        dtype=self.dtype,
                    ),
                    torch.zeros(
                        bsz,
                        self.num_heads,
                        0,
                        self.head_dim,
                        device=suffix_ids.device,
                        dtype=self.dtype,
                    ),
                )
                for _ in self.proj_kv
            ]
        x = self.embed(suffix_ids)
        return self._tokens_to_kv(x)

    def forward_full(self, input_ids: torch.Tensor) -> KVCache:
        """Reference path: entire sequence in one shot (same result as concat for this toy)."""
        x = self.embed(input_ids)
        return self._tokens_to_kv(x)


def benchmark_prefix_reuse(
    model: TinyPrefixModel, cache: PrefixKVCache, rounds: int = 50
) -> None:
    """Time repeated calls with identical prefix vs clearing the cache each time."""
    torch.manual_seed(0)
    device = model.embed.weight.device
    prefix_len = 32
    total_len = 48
    shared_prefix = torch.randint(0, 256, (1, prefix_len), device=device)
    full_a = torch.cat(
        [
            shared_prefix,
            torch.randint(0, 256, (1, total_len - prefix_len), device=device),
        ],
        dim=1,
    )
    full_b = torch.cat(
        [
            shared_prefix,
            torch.randint(0, 256, (1, total_len - prefix_len), device=device),
        ],
        dim=1,
    )

    # Warm-up
    for _ in range(3):
        _ = cache.forward_with_prefix(model, full_a, prefix_len)
        model.forward_full(full_a)

    t0 = time.perf_counter()
    for _ in range(rounds):
        _ = cache.forward_with_prefix(model, full_a, prefix_len)
        _ = cache.forward_with_prefix(model, full_b, prefix_len)
    t_cached = time.perf_counter() - t0

    fresh = PrefixKVCache(verbose=False)
    t1 = time.perf_counter()
    for _ in range(rounds):
        fresh.prefix_cache.clear()
        _ = fresh.forward_with_prefix(model, full_a, prefix_len)
        fresh.prefix_cache.clear()
        _ = fresh.forward_with_prefix(model, full_b, prefix_len)
    t_nocache = time.perf_counter() - t1

    print(f"\nBenchmark ({rounds} rounds × 2 requests, prefix_len={prefix_len}):")
    print(f"  With prefix KV cache:    {t_cached * 1000:.2f} ms")
    print(f"  Clearing cache each time: {t_nocache * 1000:.2f} ms")


def _demo() -> None:
    torch.manual_seed(42)
    # CPU keeps the toy model bitwise consistent between one-shot vs prefix+suffix paths.
    device = "cpu"
    dtype = torch.float32

    model = TinyPrefixModel(
        vocab_size=256,
        num_layers=4,
        d_model=64,
        num_heads=4,
        device=device,
        dtype=dtype,
    ).eval()

    prefix_len = 10
    total_len = 16
    shared = torch.randint(0, 256, (1, prefix_len), device=device)
    full = torch.cat(
        [shared, torch.randint(0, 256, (1, total_len - prefix_len), device=device)],
        dim=1,
    )

    cache = PrefixKVCache(verbose=True)

    print("First call (same shared prefix):")
    kv_cached_path = cache.forward_with_prefix(model, full, prefix_len)
    kv_full = model.forward_full(full)

    for i, ((ck, cv), (fk, fv)) in enumerate(zip(kv_cached_path, kv_full, strict=True)):
        # Parallel CPU matmul can reorder FP32 reductions; rtol captures that noise.
        match_k = torch.allclose(ck, fk, rtol=1e-4, atol=1e-5)
        match_v = torch.allclose(cv, fv, rtol=1e-4, atol=1e-5)
        print(
            f"  Layer {i}: concat(prefix,suffix) matches full forward: k={match_k}, v={match_v}"
        )

    print("\nSecond call with another suffix but SAME prefix:")
    full2 = torch.cat(
        [shared, torch.randint(0, 256, (1, total_len - prefix_len), device=device)],
        dim=1,
    )
    _ = cache.forward_with_prefix(model, full2, prefix_len)

    cache.verbose = False
    benchmark_prefix_reuse(model, cache)


if __name__ == "__main__":
    _demo()
