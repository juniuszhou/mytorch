# Loss Functions in PyTorch
# This file demonstrates different loss functions, their characteristics, and use cases

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("PYTORCH LOSS FUNCTIONS DEMONSTRATION")
print("=" * 80)

# ============================================================================
# 1. MEAN SQUARED ERROR (MSE) LOSS
# ============================================================================
print("\n" + "=" * 80)
print("1. MEAN SQUARED ERROR (MSE) LOSS")
print("=" * 80)
print(
    """
Use Case: Regression problems (predicting continuous values)
- House price prediction
- Temperature forecasting
- Stock price prediction
- Any continuous output prediction

Formula: MSE = mean((predicted - target)²)
Characteristics:
- Penalizes large errors more heavily (quadratic)
- Smooth gradient everywhere
- Sensitive to outliers
"""
)

mse_loss = nn.MSELoss()
# Example: Predicting house prices
predicted_prices = torch.tensor([250000.0, 300000.0, 180000.0])
target_prices = torch.tensor([240000.0, 310000.0, 175000.0])
mse_value = mse_loss(predicted_prices, target_prices)
print("Example - House Price Prediction:")
print(f"  Predicted: {predicted_prices.tolist()}")
print(f"  Target:    {target_prices.tolist()}")
print(f"  MSE Loss:  {mse_value.item():.2f}")

# ============================================================================
# 2. MEAN ABSOLUTE ERROR (L1) LOSS
# ============================================================================
print("\n" + "=" * 80)
print("2. MEAN ABSOLUTE ERROR (L1) LOSS")
print("=" * 80)
print(
    """
Use Case: Regression problems, especially when outliers are present
- Robust to outliers (less sensitive than MSE)
- Sparse solutions (encourages zero weights)
- Image denoising
- Robust regression

Formula: MAE = mean(|predicted - target|)
Characteristics:
- Linear penalty (less harsh on large errors)
- More robust to outliers than MSE
- Non-smooth at zero (gradient discontinuity)
"""
)

l1_loss = nn.L1Loss()
l1_value = l1_loss(predicted_prices, target_prices)
print("Example - Same House Price Prediction:")
print(f"  Predicted: {predicted_prices.tolist()}")
print(f"  Target:    {target_prices.tolist()}")
print(f"  L1 Loss:   {l1_value.item():.2f}")
print(
    f"  Note: L1 ({l1_value.item():.2f}) is less sensitive to outliers than MSE ({mse_value.item():.2f})"
)

# ============================================================================
# 3. CROSS ENTROPY LOSS
# ============================================================================
print("\n" + "=" * 80)
print("3. CROSS ENTROPY LOSS")
print("=" * 80)
print(
    """
Use Case: Multi-class classification problems
- Image classification (CIFAR-10, ImageNet)
- Text classification
- Handwritten digit recognition (MNIST)
- Any classification with multiple classes

Formula: CE = -log(softmax(predicted)[target_class])
Characteristics:
- Works with raw logits (no need for softmax in forward pass)
- Handles multi-class classification naturally
- Includes softmax internally
- Penalizes confident wrong predictions heavily
"""
)

ce_loss = nn.CrossEntropyLoss()
# Example: Classifying images into 3 classes (cat, dog, bird)
# Input: raw logits for 3 classes (no softmax needed!)
logits = torch.tensor(
    [
        [2.0, 1.0, 0.1],  # Predicted: class 0 (cat)
        [0.5, 2.5, 0.3],  # Predicted: class 1 (dog)
        [0.2, 0.1, 3.0],
    ]
)  # Predicted: class 2 (bird)
targets = torch.tensor([0, 1, 2])  # True labels: cat, dog, bird
ce_value = ce_loss(logits, targets)
print("Example - Image Classification (3 classes: cat, dog, bird):")
print(f"  Logits shape: {logits.shape}")
print(f"  Targets: {targets.tolist()}")
print(f"  Cross Entropy Loss: {ce_value.item():.4f}")

# Show probabilities after softmax
probs = F.softmax(logits, dim=1)
print(f"  Probabilities: {probs.tolist()}")

# ============================================================================
# 4. BINARY CROSS ENTROPY LOSS
# ============================================================================
print("\n" + "=" * 80)
print("4. BINARY CROSS ENTROPY (BCE) LOSS")
print("=" * 80)
print(
    """
Use Case: Binary classification problems
- Spam detection (spam/not spam)
- Medical diagnosis (disease/healthy)
- Sentiment analysis (positive/negative)
- Anomaly detection

Formula: BCE = -[target*log(predicted) + (1-target)*log(1-predicted)]
Characteristics:
- Requires sigmoid activation (outputs between 0 and 1)
- Handles single binary output
- Can handle multiple independent binary classifications
"""
)

bce_loss = nn.BCELoss()
# Example: Spam detection
sigmoid_outputs = torch.tensor([0.9, 0.1, 0.7])  # After sigmoid: probabilities
binary_targets = torch.tensor([1.0, 0.0, 1.0])  # True labels: spam, not spam, spam
bce_value = bce_loss(sigmoid_outputs, binary_targets)
print("Example - Spam Detection:")
print(f"  Probabilities (after sigmoid): {sigmoid_outputs.tolist()}")
print(f"  Targets: {binary_targets.tolist()}")
print(f"  BCE Loss: {bce_value.item():.4f}")

# BCE with logits (includes sigmoid internally)
bce_logits_loss = nn.BCEWithLogitsLoss()
raw_logits = torch.tensor([2.2, -2.2, 0.85])  # Raw logits (before sigmoid)
bce_logits_value = bce_logits_loss(raw_logits, binary_targets)
print("\n  Using BCEWithLogitsLoss (includes sigmoid):")
print(f"  Raw logits: {raw_logits.tolist()}")
print(f"  BCE Loss: {bce_logits_value.item():.4f}")

# ============================================================================
# 5. SMOOTH L1 LOSS (HUBER LOSS)
# ============================================================================
print("\n" + "=" * 80)
print("5. SMOOTH L1 LOSS (HUBER LOSS)")
print("=" * 80)
print(
    """
Use Case: Regression with robustness to outliers
- Object detection (bounding box regression)
- Fast R-CNN, YOLO
- When you want MSE behavior for small errors, L1 for large errors

Formula: 
  if |x| < beta: 0.5 * x² / beta
  else: |x| - 0.5 * beta
Characteristics:
- Combines benefits of MSE and L1
- Smooth everywhere (unlike L1)
- Less sensitive to outliers than MSE
- Default beta=1.0
"""
)

smooth_l1_loss = nn.SmoothL1Loss()
sl1_value = smooth_l1_loss(predicted_prices, target_prices)
print("Example - Bounding Box Regression:")
print(f"  Predicted: {predicted_prices.tolist()}")
print(f"  Target:    {target_prices.tolist()}")
print(f"  Smooth L1 Loss: {sl1_value.item():.2f}")

# ============================================================================
# 6. NEGATIVE LOG LIKELIHOOD (NLL) LOSS
# ============================================================================
print("\n" + "=" * 80)
print("6. NEGATIVE LOG LIKELIHOOD (NLL) LOSS")
print("=" * 80)
print(
    """
Use Case: Multi-class classification (when you apply softmax manually)
- Custom classification networks
- When you need explicit control over softmax
- Often used with LogSoftmax for numerical stability

Formula: NLL = -log(probabilities[target_class])
Characteristics:
- Requires log-probabilities (after LogSoftmax)
- More numerically stable than CrossEntropy in some cases
- You control the softmax application
"""
)

nll_loss = nn.NLLLoss()
# Example: Manual softmax application
log_probs = F.log_softmax(logits, dim=1)  # Apply LogSoftmax manually
nll_value = nll_loss(log_probs, targets)
print("Example - Manual Softmax Classification:")
print(f"  Log probabilities shape: {log_probs.shape}")
print(f"  Targets: {targets.tolist()}")
print(f"  NLL Loss: {nll_value.item():.4f}")
print(f"  Note: NLL ({nll_value.item():.4f}) ≈ CrossEntropy ({ce_value.item():.4f})")

# ============================================================================
# 7. KL DIVERGENCE LOSS
# ============================================================================
print("\n" + "=" * 80)
print("7. KL DIVERGENCE LOSS")
print("=" * 80)
print(
    """
Use Case: 
- Variational Autoencoders (VAEs)
- Knowledge distillation (teacher-student networks)
- Probabilistic models
- Measuring difference between probability distributions

Formula: KL(P||Q) = sum(P * log(P / Q))
Characteristics:
- Measures how one probability distribution differs from another
- Asymmetric (KL(P||Q) ≠ KL(Q||P))
- Always non-negative
- Zero when distributions are identical
"""
)

kl_loss = nn.KLDivLoss(reduction="batchmean")
# Example: Knowledge distillation
teacher_probs = F.softmax(torch.tensor([[2.0, 1.0, 0.1]]), dim=1)
student_log_probs = F.log_softmax(torch.tensor([[1.5, 1.2, 0.3]]), dim=1)
kl_value = kl_loss(student_log_probs, teacher_probs)
print("Example - Knowledge Distillation:")
print(f"  Teacher probabilities: {teacher_probs.tolist()}")
print(f"  Student log-probabilities: {student_log_probs.tolist()}")
print(f"  KL Divergence: {kl_value.item():.4f}")

# ============================================================================
# 8. MARGIN RANKING LOSS
# ============================================================================
print("\n" + "=" * 80)
print("8. MARGIN RANKING LOSS")
print("=" * 80)
print(
    """
Use Case:
- Learning to rank
- Recommendation systems
- Information retrieval
- When you want to learn relative ordering

Formula: max(0, -target * (input1 - input2) + margin)
Characteristics:
- Learns relative ordering rather than absolute values
- Used in pairwise ranking
- Margin parameter controls separation
"""
)

margin_ranking_loss = nn.MarginRankingLoss(margin=1.0)
# Example: Recommendation system (prefer item1 over item2)
input1 = torch.tensor([0.8, 0.6])  # Scores for item1
input2 = torch.tensor([0.5, 0.7])  # Scores for item2
target = torch.tensor([1, -1])  # 1: input1 > input2, -1: input1 < input2
mrl_value = margin_ranking_loss(input1, input2, target)
print("Example - Recommendation Ranking:")
print(f"  Item1 scores: {input1.tolist()}")
print(f"  Item2 scores: {input2.tolist()}")
print(f"  Target (1=item1>item2, -1=item1<item2): {target.tolist()}")
print(f"  Margin Ranking Loss: {mrl_value.item():.4f}")

# ============================================================================
# 9. COSINE EMBEDDING LOSS
# ============================================================================
print("\n" + "=" * 80)
print("9. COSINE EMBEDDING LOSS")
print("=" * 80)
print(
    """
Use Case:
- Learning embeddings
- Siamese networks
- Similarity learning
- Face recognition
- When similarity in direction matters more than magnitude

Formula: 1 - cos(input1, input2) if target=1, else max(0, cos(input1, input2) - margin)
Characteristics:
- Measures cosine similarity between vectors
- Useful when vector direction matters more than magnitude
- Margin parameter for negative pairs
"""
)

cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
# Example: Learning similar embeddings
vec1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
vec2 = torch.tensor([[1.0, 0.1], [0.1, 1.0]])  # Similar to vec1
target = torch.tensor([1, 1])  # 1: similar, -1: dissimilar
cos_value = cosine_loss(vec1, vec2, target)
print("Example - Embedding Similarity:")
print(f"  Vector1: {vec1.tolist()}")
print(f"  Vector2: {vec2.tolist()}")
print(f"  Target (1=similar, -1=dissimilar): {target.tolist()}")
print(f"  Cosine Embedding Loss: {cos_value.item():.4f}")

# ============================================================================
# 10. HINGE EMBEDDING LOSS
# ============================================================================
print("\n" + "=" * 80)
print("10. HINGE EMBEDDING LOSS")
print("=" * 80)
print(
    """
Use Case:
- Semi-supervised learning
- Learning embeddings with margin
- Similarity learning
- When you want margin-based separation

Formula: 
  if target=1: max(0, margin - input)
  if target=-1: max(0, input - margin)
Characteristics:
- Margin-based loss
- Encourages separation between classes
- Used in embedding learning
"""
)

hinge_loss = nn.HingeEmbeddingLoss(margin=1.0)
hinge_input = torch.tensor([0.5, 1.5, -0.5])
hinge_target = torch.tensor([1, 1, -1])
hinge_value = hinge_loss(hinge_input, hinge_target)
print("Example - Margin-based Embedding:")
print(f"  Input: {hinge_input.tolist()}")
print(f"  Target: {hinge_target.tolist()}")
print(f"  Hinge Embedding Loss: {hinge_value.item():.4f}")

# ============================================================================
# VISUALIZATION: Comparing Loss Functions
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZING LOSS FUNCTION BEHAVIORS")
print("=" * 80)


def visualize_loss_functions():
    """Create diagrams comparing different loss functions"""

    # Create error values (difference between predicted and target)
    errors = np.linspace(-3, 3, 1000)
    errors_tensor = torch.tensor(errors, dtype=torch.float32)
    zero_target = torch.zeros_like(errors_tensor)

    # Calculate losses
    mse_values = nn.MSELoss(reduction="none")(errors_tensor, zero_target).numpy()
    l1_values = nn.L1Loss(reduction="none")(errors_tensor, zero_target).numpy()
    smooth_l1_values = nn.SmoothL1Loss(reduction="none")(
        errors_tensor, zero_target
    ).numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Comparison of Regression Loss Functions", fontsize=16, fontweight="bold"
    )

    # Plot 1: MSE vs L1 vs Smooth L1
    ax1 = axes[0, 0]
    ax1.plot(errors, mse_values, label="MSE Loss", linewidth=2, color="blue")
    ax1.plot(errors, l1_values, label="L1 Loss", linewidth=2, color="red")
    ax1.plot(
        errors, smooth_l1_values, label="Smooth L1 Loss", linewidth=2, color="green"
    )
    ax1.set_xlabel("Error (predicted - target)", fontsize=11)
    ax1.set_ylabel("Loss Value", fontsize=11)
    ax1.set_title("Regression Loss Functions", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 3)

    # Plot 2: Classification Loss (Cross Entropy)
    ax2 = axes[0, 1]
    # Simulate binary classification: probability vs loss
    probs = np.linspace(0.01, 0.99, 1000)
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    target_ones = torch.ones_like(probs_tensor)
    target_zeros = torch.zeros_like(probs_tensor)

    bce_loss_ones = nn.BCELoss(reduction="none")(probs_tensor, target_ones).numpy()
    bce_loss_zeros = nn.BCELoss(reduction="none")(probs_tensor, target_zeros).numpy()

    ax2.plot(
        probs, bce_loss_ones, label="BCE Loss (target=1)", linewidth=2, color="purple"
    )
    ax2.plot(
        probs, bce_loss_zeros, label="BCE Loss (target=0)", linewidth=2, color="orange"
    )
    ax2.set_xlabel("Predicted Probability", fontsize=11)
    ax2.set_ylabel("Loss Value", fontsize=11)
    ax2.set_title("Binary Cross Entropy Loss", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    # Plot 3: Cross Entropy for Multi-class
    ax3 = axes[1, 0]
    # Simulate 3-class classification
    logit_diffs = np.linspace(
        -5, 5, 1000
    )  # Difference between correct and wrong class logits
    correct_logit = torch.tensor(logit_diffs, dtype=torch.float32).unsqueeze(1)
    wrong_logit = torch.zeros_like(correct_logit)
    logits_3class = torch.cat([correct_logit, wrong_logit, wrong_logit], dim=1)
    targets_3class = torch.zeros(len(logit_diffs), dtype=torch.long)

    ce_loss_3class = nn.CrossEntropyLoss(reduction="none")(
        logits_3class, targets_3class
    ).numpy()

    ax3.plot(
        logit_diffs,
        ce_loss_3class,
        label="Cross Entropy Loss",
        linewidth=2,
        color="darkblue",
    )
    ax3.set_xlabel("Logit Difference (correct - wrong)", fontsize=11)
    ax3.set_ylabel("Loss Value", fontsize=11)
    ax3.set_title("Multi-class Cross Entropy Loss", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5, 5)

    # Plot 4: Loss Function Comparison Table (text-based visualization)
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create a summary table
    loss_info = [
        ["Loss Function", "Use Case", "Key Property"],
        ["MSE", "Regression", "Quadratic penalty"],
        ["L1", "Regression", "Robust to outliers"],
        ["Smooth L1", "Regression", "Smooth, robust"],
        ["Cross Entropy", "Multi-class", "Includes softmax"],
        ["BCE", "Binary class", "Requires sigmoid"],
        ["NLL", "Multi-class", "Manual softmax"],
        ["KL Divergence", "VAEs, Distillation", "Distribution distance"],
    ]

    table = ax4.table(
        cellText=loss_info[1:],
        colLabels=loss_info[0],
        cellLoc="left",
        loc="center",
        colWidths=[0.25, 0.4, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the header
    for i in range(len(loss_info[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax4.set_title(
        "Loss Function Quick Reference", fontsize=12, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig("loss_functions_comparison.png", dpi=300, bbox_inches="tight")
    print("Visualization saved as 'loss_functions_comparison.png'")
    plt.show()


# Generate visualizations
try:
    visualize_loss_functions()
except Exception as e:
    print(f"Note: Visualization requires matplotlib. Error: {e}")
    print("Install with: pip install matplotlib")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("LOSS FUNCTION SELECTION GUIDE")
print("=" * 80)
print(
    """
┌─────────────────────┬──────────────────────────┬─────────────────────────────┐
│ Loss Function       │ Primary Use Case         │ Key Characteristics          │
├─────────────────────┼──────────────────────────┼─────────────────────────────┤
│ MSE Loss            │ Regression               │ Quadratic penalty           │
│ L1 Loss             │ Regression (robust)      │ Linear penalty, robust      │
│ Smooth L1 Loss      │ Regression (balanced)     │ Smooth, less outlier sens.  │
│ Cross Entropy       │ Multi-class classification│ Includes softmax            │
│ BCE Loss            │ Binary classification     │ Requires sigmoid            │
│ BCEWithLogitsLoss   │ Binary classification     │ Includes sigmoid            │
│ NLL Loss            │ Multi-class (manual)      │ Requires LogSoftmax         │
│ KL Divergence       │ VAEs, Knowledge Distill. │ Measures dist. difference   │
│ Margin Ranking      │ Learning to rank         │ Pairwise comparison         │
│ Cosine Embedding    │ Similarity learning      │ Direction-based similarity  │
│ Hinge Embedding     │ Embedding learning       │ Margin-based separation     │
└─────────────────────┴──────────────────────────┴─────────────────────────────┘

Quick Decision Tree:
1. Classification or Regression?
   - Classification → Go to 2
   - Regression → Go to 3
   
2. Classification Type:
   - Binary → Use BCEWithLogitsLoss
   - Multi-class → Use CrossEntropyLoss
   
3. Regression Type:
   - Standard → Use MSELoss
   - Outliers present → Use L1Loss or SmoothL1Loss
   - Object detection → Use SmoothL1Loss
"""
)

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)
