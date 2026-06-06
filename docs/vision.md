# vision

## course 

https://www.youtube.com/watch?v=tr-CUpw--ck&t=4127s



## multi modal
一句话总结：
Autoregressive 是“一边想一边写”，适合统一建模和复杂逻辑；
Diffusion 是“从模糊到清晰反复打磨”，适合生成高保真视觉内容。

目前最强多模态生成系统，基本都是两者结合的产物。未来很可能是 LLM（AR）做大脑，Diffusion 做手 的架构。

Diffusion 模型的核心是去噪网络（Denoising Network），只要这个网络能预测噪声（或 score）就行。
早期/经典 Diffusion 都用 U-Net（CNN 架构）

现代 Diffusion 大量转向 Transformer（DiT）
从 2023 年开始，Diffusion Transformer (DiT) 成为主流趋势：

