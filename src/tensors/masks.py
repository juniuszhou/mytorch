"""demo different mask operations"""

import torch


# return lower triangular matrix as 1, then higher part will be 0
def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
        1, 1, seq_len, seq_len
    )


def mask_demo():
    x = torch.randn(6, 6).cuda()
    print("x", x)

    mask = generate_causal_mask(x.size(1), x.device)

    x = x.masked_fill(mask == 0, float("-inf"))
    print(x)


if __name__ == "__main__":
    mask_demo()
