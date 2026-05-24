# use the simple model to try distributed parallel

## start DP

just one process per node. DP will use the multiple thread for parallelism.

```shell
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=101 --rdzv_endpoint="localhost:5972" simple_model_dp.py
```

## start DDP

the parameter is the same since we simulate the multiple process in the same node.

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint="localhost:5972" simple_model_ddp.py
```

## start FSDP2

```shell
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint="localhost:5972" simple_model_fsdp2.py

torchrun --nnodes=1 --nproc_per_node=2 simple_model_fsdp2.py
```

## start TP

```shell
torchrun --nnodes=1 --nproc_per_node=2 simple_model_tp.py
```

| Aspect                          | DataParallel (DP)                                           | DistributedDataParallel (DDP)                              | Winner (2025–2026)            |
| ------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------- |
| Parallelism type                | Single-process, multi-threaded                              | Multi-process (one process per GPU)                        | **DDP**                       |
| Number of machines              | Single machine only                                         | Single machine **or** multi-node                           | **DDP**                       |
| Speed (same machine)            | Usually slower (20–50%+ overhead common)                    | Faster — better GPU utilization, lower overhead            | **DDP**                       |
| Scalability                     | Poor — bottlenecks severely beyond 2–4 GPUs                 | Excellent — scales well to 8, 16, 32+ GPUs, multi-node     | **DDP**                       |
| Gradient synchronization        | All gradients gathered to main thread, then broadcast       | Each process computes grads → `all_reduce` to average      | **DDP**                       |
| Python GIL impact               | High — serious contention between threads                   | None — separate processes                                  | **DDP**                       |
| Model replication               | Replicates model **every forward pass** (large overhead)    | Model copied once per process at initialization            | **DDP**                       |
| Ease of use                     | Very easy — one line: `nn.DataParallel(model)`              | More setup: process group init, `torchrun`, rank handling  | **DP** (only for quick tests) |
| Custom forward / complex models | Frequently breaks (multiple outputs, custom autograd, etc.) | Very robust — almost identical to single-GPU code          | **DDP**                       |
| Batch size behavior             | Pass global batch size — DP auto-splits it                  | Pass **local** batch size — global = local × world_size    | —                             |
| Official PyTorch recommendation | Legacy — still functional but **discouraged**               | **Recommended** for nearly all multi-GPU use-cases         | **DDP**                       |
| Still useful in 2025–2026?      | Only for very quick 1–2 GPU prototyping                     | Default choice for serious / production multi-GPU training | **DDP**                       |
