# FPO++ Code Release

[[project page](https://hongsukchoi.github.io/fpo-control/)] [[arxiv](https://arxiv.org/abs/2602.02481)]

This repository contains experiments for FPO++ (Flow Policy Optimization), which have been cleaned up and refactored for release.

<table><tr><td>
  <strong>Flow Policy Gradients for Robot Control.</strong><br />
  <small></small>Brent&nbsp;Yi<sup>12*</sup>, Hongsuk&nbsp;Choi<sup>12*</sup>, Himanshu&nbsp;Gaurav&nbsp;Singh<sup>12</sup>, Xiaoyu&nbsp;Huang<sup>12</sup>, Takara&nbsp;E.&nbsp;Truong<sup>13</sup>, Carmelo&nbsp;Sferrazza<sup>1</sup>, Yi&nbsp;Ma<sup>24</sup>, Rocky&nbsp;Duan<sup>1†</sup>, Pieter&nbsp;Abbeel<sup>12†</sup>, Guanya&nbsp;Shi<sup>15†</sup>, Karen&nbsp;Liu<sup>13†</sup>, and&nbsp;Angjoo&nbsp;Kanazawa<sup>12&dagger;</sup><small>

</td></tr>
</table>
<sup>1</sup><em>Amazon FAR</em>, <sup>2</sup><em>UC Berkeley</em>, <sup>3</sup><em>Stanford</em>, <sup>4</sup><em>HKU</em>, <sup>5</sup><em>CMU</em>, <sup>*</sup><em>Equal Contribution</em>, <sup>†</sup><em>Amazon FAR Team Co-lead</em>

## Directory Structure

- **[`isaaclab_experiments/`](isaaclab_experiments/)** — Isaac Lab experiments: velocity-conditioned locomotion (6+ robots). Includes training commands and expected training curves.
- **[`manipulation_experiments/`](manipulation_experiments/)** — Manipulation experiments: pretraining and fine-tuning across five manipulation tasks using FPO++, Vanilla FPO, and DPPO variants. Includes training commands and expected training curves.

Each directory is self-contained with its own `setup_env.sh`, `source_env.sh`, and README with setup/usage instructions.

## Setup

Each experiment directory has independent dependencies and conda environments. See the README in each directory for detailed setup instructions. In brief:

```bash
# Isaac Lab experiments
cd isaaclab_experiments
bash setup_env.sh        # one-time setup
source source_env.sh     # activate env (each session)

# Manipulation experiments
cd manipulation_experiments
bash setup_env.sh        # one-time setup
source source_env.sh     # activate env (each session)
```

## Git Submodules

This repository uses git submodules for Isaac Lab extension (under `isaaclab_experiments/`). After cloning, initialize them with:

```bash
git submodule update --init --recursive
```

This is also done automatically by `isaaclab_experiments/setup_env.sh`.

## Licenses

This repository includes and adapts code from the following third-party projects. Original license files and copyright headers are retained in all cases.

### Isaac Lab Experiments (`isaaclab_experiments/`)

| Project | License | Inclusion | What we use/adapt |
|---------|---------|-----------|-------------------|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause | Git submodule | Simulation framework, task definitions, robot assets |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | BSD-3-Clause | Adapted (not vendored) | Actor-critic modules, on-policy runner, rollout storage, normalizer |

The `isaaclab_fpo` package adapts code from rsl_rl and IsaacLab (VecEnv wrapper, config dataclasses, training/evaluation scripts). Original copyright headers are retained in all adapted files.

### Manipulation Experiments (`manipulation_experiments/`)

| Project | License | Inclusion | What we use/adapt |
|---------|---------|-----------|-------------------|
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Vendored under `thirdparty/lerobot/` (minor modifications, see `CHANGES.txt`) | Policy architectures and training utilities |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT | Vendored under `thirdparty/robosuite/` (minor modifications, see `CHANGES.txt`) | Simulation environments for manipulation tasks |
| [DPPO](https://github.com/irom-lab/dppo) | MIT | Adapted file (`src/vit.py`) | Vision Transformer encoder |
| [DexMimicGen](https://github.com/NVlabs/dexmimicgen) | NVIDIA Source Code License (non-commercial) | Cloned at setup time | Dexterous manipulation data generation |

**核心区别：`FSPPOJoint` 的 `aux_loss` 是额外的回归梯度；`FPO` 的 CFM loss 则用于构造策略比率，是策略梯度的计算路径。**

| | FSPPOJoint `aux_loss` | FPO `cfm loss` |
|---|---|---|
| 进入更新的方式 | `PPO surrogate + λ_aux × pMF loss` | `ρ = exp(CFM_old − CFM_current)`，再用 `ρ` 计算 PPO/SPO/ASPO surrogate |
| 优势 \(A\) 的作用 | aux 梯度**不乘 \(A\)**，对好坏动作都做回归 | \(A\) 决定方向：在未裁剪且 \(\rho\approx1\) 时，正优势降低该动作的 CFM loss，负优势提高它 |
| 对 actor 的影响 | 与 PPO 梯度相加，可能协同，也可能冲突；PPO 比率不包含 aux | CFM 梯度就是该 surrogate 传到 actor 的梯度路径 |

具体地，FSPPOJoint 用保存的 latent、动作和条件高斯 log-prob 计算联合策略的 PPO 比率；`aux_loss` 另用 rollout 动作及新采样的噪声、时间计算 pMF 的 \(u/v\) 回归误差，并在同一次 `backward()` 中更新共享 actor。[fsppo_joint.py](/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/algorithms/fsppo_joint.py:463) [actor_critic.py](/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py:1028) 因此，即使某些样本的 PPO 梯度被裁剪，aux 仍可能推动 actor；它也会参与全模型梯度范数裁剪和候选步预算检查。

FPO **没有单独的 `+ λ × mean(CFM loss)` 项**。对未触发裁剪的样本，最小化的 surrogate \(L=-A\rho\) 满足 \(\nabla_\theta L=A\rho\nabla_\theta\mathrm{CFM}_{current}\)。实际的 PPO/ASPO 分支和 CFM 裁剪会改变这一局部关系；这里的 CFM 比率也只是分数比替代量，并非精确动作概率比。[fpo.py](/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/algorithms/fpo.py:529)

**当前默认 `fsppo_joint_aux_loss_coef=0`，所以 aux 完全不产生梯度。**[rl_cfg.py](/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/rl_cfg.py:413) 你工作区中的[校验代码](/root/R2A/Algos/fpo-control-saa/isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/algorithms/fsppo_joint.py:103)已注释掉非负检查：若设负系数，会对**所有**样本反向优化 pMF 回归，不能等同于 FPO 对负优势动作的更新。

以上是当前源码的梯度路径分析；两个 loss 的标量大小不能直接代表各自对策略更新的贡献，实际比较应看 actor 上两项梯度的范数及夹角。



