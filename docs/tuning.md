# Tuning

## 一般模型的训练

- Pre-training（预训练）
  目标：让模型学会语言的基本统计规律、语法、世界知识
  数据：海量无监督文本（万亿 ~ 几十万亿 token）
  目标函数：Next Token Prediction（自回归语言建模）
  这是最耗资源的部分（几千 ~ 几万 H100/A100 天），产出的是 Base 模型（像 LLaMA-3-8B base、Qwen2-7B base）
  → 这个阶段严格来说不算“tuning”，而是 from scratch 训练，但它是所有后续 tuning 的基础。

- Supervised Fine-Tuning（SFT） / Instruction Fine-Tuning（指令微调）
  目标：让模型学会“听懂人类指令 + 按格式输出高质量回答”
  数据：高质量的 instruction-response 对（Alpaca、ShareGPT、UltraChat、OpenHermes、合成数据等）
  目标函数：还是 Next Token Prediction，但只在 curated 的对话/任务数据上继续训练
  产出：Instruct / Chat 模型 的初步版本（像 LLaMA-3-8B-Instruct 第一版）
  → 这是大多数人说的“fine-tuning”的起点，消耗资源少很多（几张卡几天就能出结果）

- Alignment / Preference Tuning（对齐 / 偏好优化）
  目标：让模型输出更符合人类偏好（helpful、honest、harmless），减少 hallucination、毒性、偏见
  主流子方法（2025–2026 年已高度多样化）：

  RLHF（经典）：Reward Model + PPO（ Proximal Policy Optimization ）
  DPO（Direct Preference Optimization）：直接用偏好对优化，最流行（简单、无需 reward model、无需在线采样）
  ORPO（Odds Ratio Preference Optimization）：把 SFT + 偏好对齐合并成一步
  KTO / SimPO / GRPO 等变体

数据：人类（或 AI）标注的 preference pairs（chosen vs rejected 回复）
产出：最终的 对齐版模型（像 ChatGPT、Claude、Grok、LLaMA-3-Instruct 最终版）

## special version LLM like reasoning model

Pre-training（基础预训练，同上）

- Continued Pre-training / Mid-training / Domain-specific Pre-training（继续预训练 / 中间训练）
  在特定领域数据（代码、数学、长上下文、中文等）上继续 next-token 训练
  常用来做“知识注入”或“上下文长度扩展”

Supervised Fine-Tuning (SFT / IFT)（指令微调，同上）

- Reasoning / Thinking Tuning（可选，针对 reasoning 模型）
  用 chain-of-thought、process supervision、verifiable reward 数据强化思考过程
  常见于 o1-style 模型（RLVR、GRPO 等）

Preference / Alignment Tuning（DPO / ORPO / RLHF / RLVR 等）

- Safety / Red-teaming Tuning（可选最后一步）
  专门针对 jailbreak、敏感话题、安全性再做一轮 SFT 或 preference tuning

## tuning types

### PEFT parameter efficient fine tuning

adapt large models by training a small number of additional parameters while keeping the base model frozen.
原来的模型不动，只对一些参数进行tuning。

### LoRA

LoRA 是 PEFT 的一种
LoRA has become the most widely adopted PEFT method

It works by adding small rank decomposition matrices to the attention weights, typically reducing trainable parameters by about 90%.

LoRA adapter（客服、代码、法律、医疗等）
是的，每个 LoRA adapter 通常是用不同的数据集（或同一数据集的不同部分/不同目标）独立训练出来的结果。

### OLoRA

OLoRA utilizes QR decomposition to initialize the LoRA adapters.
OLoRA 的核心创新在于 LoRA adapter（A 和 B 矩阵）的初始化方式：它使用 QR 分解（或类似正交化过程）来初始化低秩矩阵，使其满足 正交（orthonormal） 属性，而不是标准 LoRA 的默认初始化（A 用 Kaiming-uniform 随机，B 初始化为 0）。

OLoRA 初始化 adapter 的最大卖点就是“更快收敛 + 更高天花板 + 更稳定”，几乎是零成本的升级（计算开销忽略不计）。

## Preference Tuning

### DPO direct preference optimization

把偏好数据直接转成分类损失，最简单最稳

### KTO Kahneman-Tversky Optimization

数据只有好和坏的评价，没有二个数据之间的比较

### GRPO / RLVR Reinforcement Learning with Verifiable Rewards 针对reasoning模型

deterministic & verifiable
典型奖励例子（通常是 0/1 binary，或带格式分）：
数学：最终 boxed 答案是否与标准答案字符串匹配（或用 sympy 执行验证）
代码：生成的代码在 hidden test cases 上是否全部通过
逻辑 puzzle / GSM8K / AIME：答案对错
格式遵守：是否严格遵循 <think>…</think><answer>…</answer> 结构
多步工具调用：是否最终调用正确 API 并得到预期输出
优点：
无需昂贵的人类偏好标注或训练 Reward Model
奖励信号干净、无噪声 → 训练更稳定
可无限自生成数据（采样 → 验证 → 继续）

缺点：
只适用于有明确对错的领域（math/code/logic），不适合 open-ended chat、创意写作、安全对齐
