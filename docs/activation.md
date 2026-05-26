# The Five Essential Activations

1. **Sigmoid**: Maps to (0, 1) - perfect for probabilities
2. **ReLU**: Removes negatives - creates sparsity and efficiency
3. **Tanh**: Maps to (-1, 1) - zero-centered for better training
4. **GELU**: Smooth ReLU - modern choice for advanced architectures
5. **Softmax**: Creates probability distributions - converts values to probabilities
6. SwiGLU

## variant ReLU for dying ReLU problem.

- LeakyReLU: f(x) = max(0.01\*x, x) - allows a small signal even for negative inputs
- PReLU: Adjustable slope for negative values (a learnable parameter controls the slope)
