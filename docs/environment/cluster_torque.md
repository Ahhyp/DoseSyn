# Torque 集群使用说明（DoseSyn）

> 记录时间：2026-10-24　|　登录/管理节点：`mgt`　|　调度器：Torque 6.1.0　|　队列：`gentai`
>
> 本文只记录环境事实与提交方式；DoseSyn 当前仍处于 M3 完成、M4 未开始的状态，本文件不触发任何训练任务。

## 1. 节点与 GPU

| 节点 | 资源（按管理员说明） | 用途 |
| --- | --- | --- |
| `node1` | 2 × V100 | GPU 训练 |
| `node2` | 1 × V100 | GPU 训练（记录时状态 `down`） |
| `node3` | 64 核 CPU | CPU 预处理 / 非 GPU 任务 |
| `node4` | 2 × RTX 4090 | GPU 训练 |
| `node5` | 1 × RTX 4090 | GPU 训练 |

登录节点是 **管理节点**，不要在上面直接跑长时间训练；训练任务通过 PBS 提交。

## 2. Torque 客户端路径

Torque 命令**不在默认 `PATH`** 中，需要先加入：

```bash
export PATH=/opt/software/torque-6.1.0/bin:$PATH
```

常用命令：

```bash
qnodes                 # 查看节点状态、负载、作业数
qstat -q               # 查看队列
qstat -n1              # 查看排队/运行作业及节点占用
qsub job.pbs           # 提交
qdel <jobid>           # 删除
```

## 3. 提交前确认 GPU 是否空闲

Torque 当前**没有配置 `gpus` 资源**：`qnodes` 输出中的 `gres=` 为空，队列也不感知显卡型号。因此“提交成功”不等于“分到了空闲 GPU”，必须同时确认：

1. `qnodes` 中目标节点 `state` 为 `free` 或非独占；
2. `qstat -n1` 看该节点是否已有他人的作业；
3. 进入作业后执行 `nvidia-smi`，确认显存占用与进程；
4. 通过 `CUDA_VISIBLE_DEVICES` 显式绑定空闲卡。

示例：

```bash
export PATH=/opt/software/torque-6.1.0/bin:$PATH
qnodes | grep -A3 '^node'
qstat -n1
```

记录时快照（会变化，仅作参考）：

- `node1`：`free`，3 个作业，72 核，负载 0.32；
- `node2`：`down`；
- `node3`：`free`，3 个作业，64 核，负载 3.57；
- `node4`：`free`，2 个作业，16 核，负载 7.40；
- `node5`：`job-exclusive`，1 个作业占满；
- 队列 `gentai`：运行 9、排队 3。

## 4. PBS 模板

### 4.1 CPU 任务（node3）

```bash
#!/bin/bash
#PBS -N dosesyn_cpu
#PBS -q gentai
#PBS -l nodes=1:ppn=16
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o logs/pbs_cpu.$PBS_JOBID.log

cd "$PBS_O_WORKDIR"
export PATH=/opt/software/torque-6.1.0/bin:$PATH
.venv/bin/python scripts/03_verify_splits.py
```

### 4.2 GPU 任务（整节点申请，避免与他人抢卡）

Torque 不追踪 GPU，最稳妥的方式是**申请整节点**，再在作业内确认显卡：

```bash
#!/bin/bash
#PBS -N dosesyn_gpu
#PBS -q gentai
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o logs/pbs_gpu.$PBS_JOBID.log

cd "$PBS_O_WORKDIR"
export PATH=/opt/software/torque-6.1.0/bin:$PATH

nvidia-smi
# 若节点有 2 张卡且都空闲：
export CUDA_VISIBLE_DEVICES=0,1
.venv/bin/python your_train_entry.py
```

> 若多人共享同一 GPU 节点，请先在作业内 `nvidia-smi` 查看占用，只使用空闲卡，并在实验记录中写明卡号；不要默认占用 0 号卡。

### 4.3 交互式任务

```bash
export PATH=/opt/software/torque-6.1.0/bin:$PATH
qsub -I -q gentai -l nodes=1:ppn=8 -l walltime=04:00:00
```

进入节点后再执行 `nvidia-smi` 并设置 `CUDA_VISIBLE_DEVICES`。

## 5. 与 DoseSyn 的关系

- 项目路径：`/home/kouzheng_2/AH_YeP/DoseSyn`
- Python 环境：`.venv`（Python 3.10、pandas 1.3.5、torch 1.13.1、PyG 2.3.1、RDKit 2023.03.2）
- M3 已完成：Primary-v1 与 S1/S2/S3 划分，**未训练任何网络**。
- M4 若开始训练，提交前必须按第 3 节确认空闲 GPU；不要直接登录节点训练，也不要在未确认时占用整卡。
- 大数据与中间产物在 `data/`、`outputs/`，不进入 Git；Git 只保留源码、配置、测试与报告。

## 6. 备注

- `qnodes` 里的 `jobs=...` 是已分配核数区间，不直接等于 GPU 数；GPU 占用以作业内 `nvidia-smi` 为准。
- `max_user_run = 5`、`max_user_queuable = 10`，同一用户同时运行/排队任务有上限。
- 节点状态会变化，本文快照仅用于说明格式；每次提交前重新执行 `qnodes` / `qstat -n1`。
