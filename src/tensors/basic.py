import torch


def tensor_demo():
    a = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
        device=torch.device("cuda:0"),
        dtype=torch.float16,
    )

    print("a", a.transpose(0, 1))

    a.retain_grad()
    b = a * 2
    b.retain_grad()

    print("b", a.matmul(b))

    c = (a + 1 + b) * 2
    c.retain_grad()

    print("shape of sum", c.sum().shape)
    c.sum().backward()
    print("c.grad", c.grad)
    print("a.grad", a.grad)
    print("b.grad", b.grad)


if __name__ == "__main__":
    tensor_demo()
