对，你抓到了 pMF + Policy Gradient 中一个非常关键的问题。

如果把 pMF 的 MeanFlow Loss（你说的 `u_loss`）完全去掉，只保留：

$$ \mathcal L\_{\mathrm{PG}} = -\mathbb E\_{s,\epsilon} \left[Q\_\psi(s,F\_\theta(s,\epsilon))\right] $$

那么从训练机制上看，它本质上就退化成一个单步隐式生成策略（Implicit Generative Policy），而不再是受到 Flow Matching 约束的 Flow Policy。

但这里需要区分两个概念：

- 模型能否表示 Flow 的终点映射： 可以。在适当条件下，一个映射可以由连续的流过程实现。
- 模型是否真的学习了符合 MeanFlow identity 的流过程： 不一定。仅使用 Policy Gradient，没有约束中间时间的速度场和一致性。

所以，你的直觉基本正确：没有 Flow Loss，就失去了 pMF 作为 Flow 学习方法最重要的结构性约束。

而这恰好影响我们之前讨论的 pMF-FPO 应该怎么设计。

## 1. 严格来说，pMF 优化的不是 `u_loss`

我重新核对了 pMF 原论文 第 4.3 节和官方实现。

pMF 的网络输出是 $x\_\theta$，由此定义平均速度：

$$ u\_\theta(z_t,r,t,s) = \frac{z_t-x\_\theta(z_t,r,t,s)}{t} $$

真正的 MeanFlow Identity 是：

$$ \boxed{ v(z_t,t,s) = u(z_t,r,t,s) + (t-r)\frac{d}{dt}u(z_t,r,t,s) } $$

其中：

$$ \frac{d}{dt}u = \partial_tu+ \nabla_z u\cdot v $$

pMF 根据这个关系构造：

$$ V\_\theta=u\_\theta+(t-r)\operatorname{sg}(\mathrm{JVP}) $$

然后优化：

$$ \boxed{ \mathcal L\_{\mathrm{pMF}} = \mathbb E\left[ \\|V\_\theta-(\epsilon-a)\\|^2 \right] } $$

所以它实际上是 x-prediction + v-space loss，而不是简单的 $u$-MSE。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



这项 Loss 才是连接一步生成映射与连续时间 Flow 动力学的关键。

## 2. 如果删除它，会发生什么？

假设网络只在 $r=0,t=1$ 处接受 RL 梯度：

$$ a\_\theta=x\_\theta(\epsilon,0,1,s) $$

然后优化：

$$ \max\_\theta \mathbb E[Q(s,a\_\theta)] $$

网络只需要让这一个输入切片产生高回报动作。

但对于其他时间：

$$ x\_\theta(z_t,r,t,s),\quad 0\<r\le t<1 $$

Policy Gradient 没有提供直接的学习目标。

因此，虽然你仍可以形式上定义 $u\_\theta$，但不能保证它是某个一致 Flow 的平均速度场，也不能保证它与网络的一步终点预测相匹配。

尤其是，如果模型从随机初始化开始，仅用 Policy Gradient 训练，输入中的 $r,t$ 可能完全失去实际意义。

这就解释了为什么去掉 MeanFlow Loss 后，使用 pMF 网络结构和直接使用普通 MLP Actor，在方法本质上可能没有明显差异。

## 3. 如何让 Policy Gradient 真正优化 Flow？

我认为应该保留 MeanFlow 约束，同时引入策略改进：

$$ \boxed{ \min\_\theta \underbrace{\mathcal L\_{\mathrm{PG}}(\theta)}\_{\text{提高回报}} + \lambda \underbrace{\mathcal L\_{\mathrm{pMF}}(\theta)}\_{\text{学习并维持 Flow}} } $$

但直接加权求和还有一个问题：

Flow Loss 应该拟合哪个动作分布？

如果始终使用旧策略产生的数据，MeanFlow Loss 就可能阻止策略改进。

如果只使用当前策略生成的数据，又必须认真处理不断变化的目标分布，不能认为自生成数据的低训练误差就保证了有意义的策略优化。

我更建议采用交替优化：

当前 pMF Policy

$a=x\_{\theta_k}(\epsilon,0,1,s)$

Policy Improvement

利用 Critic 或 Advantage 构造改进后的目标动作分布

pMF Projection

对改进后的动作分布做 MeanFlow 训练

新的 pMF Policy

保留一步生成，同时学习更新后的概率流

这样每轮都有明确的 Flow 学习目标。

当然，这会使我们重新接近刚才讨论的 PMD + Flow Matching 路线。若坚持直接 Policy Gradient，则需要研究如何在更新网络参数时维持 Flow 一致性，而不是完全靠后续分布投影修复。

## 4. 一个很值得做的验证实验

我建议增加一个 One-step / Multi-step Consistency Test。

固定相同状态 $s$ 和噪声 $\epsilon$，比较：

一步 pMF 输出：

$$ a\_{\mathrm{one}} = x\_\theta(\epsilon,0,1,s) $$

和由瞬时速度场积分得到的输出：

$$ a\_{\mathrm{ODE}} = \epsilon-\int_0^1 v\_\theta(z_t,t,s)\\,dt $$

其中 $v\_\theta(z_t,t,s)=u\_\theta(z_t,t,t,s)$，积分轨迹由对应 ODE 决定。

定义：

$$ \boxed{ \mathcal E\_{\mathrm{flow}} = \mathbb E\_{s,\epsilon} \left[ \\|a\_{\mathrm{one}}-a\_{\mathrm{ODE}}\\|^2 \right] } $$

你可以比较三个版本：

| 方法             | PG Loss | MeanFlow Loss |
| -------------- | ------- | ------------- |
| A：仅 PG         | 有       | 无             |
| B：pMF + PG     | 有       | 有             |
| C：交替 PMD + pMF | PMD 改进  | 有             |

在相同噪声和状态下观察回报、多模态动作覆盖以及 $\mathcal E\_{\mathrm{flow}}$。这个误差是实用的一致性诊断，不等同于完整的分布匹配保证。

## 我的结论

如果你想研究的是 pMF-FPO，而不只是 One-step Generative PPO，那么 MeanFlow 的动力学关系就应该成为算法的一部分。

但不一定非要保留原始的 pMF Loss。理论上也可以用其他有效的 Flow Matching、Flow Map Consistency 或动力学约束来实现。

我认为下一步最重要的是研究：

$$ \boxed{ \begin{gathered} \text{Policy Gradient 如何更新}\\\ \text{MeanFlow 的平均速度场 }u\_\theta\\\ \text{同时保持一步生成与 Flow 一致性？} \end{gathered} } $$

这比单纯把 `loss_actor = -Q(s,a)` 加进 pMF，更接近一篇真正以 Flow Policy Optimization 为核心的论文。

## 当前验证实现

可执行的 A/B 训练开关、检查点评估、共享状态/噪声 bank、两种瞬时场 ODE 诊断和运行命令见
[pMF Flow 验证说明](/root/space_robotics_bench_l/analysis/pmf_flow_validation.md)。
入口是 `/root/space_robotics_bench_l/eval_isaacbase_pmf.py`，沿用
`train_isaacbase_fsppo_joint.py` 的主要启动路径。
`diagonal_mean` 严格对应本页 `v=u(z,t,t)`；`auxiliary` 另行诊断当前实现的 JVP 速度头。
C 的交替 PMD+pMF 尚未实现，不将现有 advantage 加权回归标记为 C。


## Next Step

### 本轮代码落地（2026-10-11）

已增加独立 `PMFPMD`（新增 action-conditioned Q、ESS 候选重加权、纯 pMF 投影、
可选平方 Gaussian map-KL trust region）和 `FSPPOJointAligned` A/D 对照，
没有把 PPO + PMD + pMF 三种 actor loss 叠加。
SRB 统一入口为 `train_isaacbase_pmf_next.py --variant joint_pg|pmd|pmd_tr|joint_pmf`，
共同配置 `hyperparams/pmf_next_step.yaml`，默认先对齐15M环境步、256×24 rollout。
另有 frozen PPO RAW state collector、完整动作协方差分解和普通 pMF teacher distillation。
实现细节、全部命令、Q/trust-region的限制与验证范围见
[pmf_next_step.md](/root/space_robotics_bench_l/analysis/pmf_next_step.md)。
这是一版可执行实验原型；CPU 合约验证不代表真实 H1 GPU 收益。

我认为应该尝试 PMD，但现在不建议直接在 pMF-PPO 上再叠加一个 PMD Loss。

看完你上传的 Flat-H1 消融实验报告，我的判断是：

目前的主要问题不一定是 PPO 不适合 pMF，而是 pMF 的策略改进机制和 MeanFlow 生成机制还没有有效地结合起来。

PMD 可以成为解决这个问题的一条路线，但我建议分两步：

1. 先排查为什么 pMF-PPO 的基本策略学习明显慢于 Gaussian PPO。
2. 再尝试 PMD-driven pMF：由 PMD 产生改进后的目标动作分布，用 pMF Loss 学习它。

## 一、你的实验数据说明了什么？

从报告中提取几个关键指标：

| 指标           | Gaussian PPO | pMF PG-only | pMF PG+pMF |
| ------------ | ------------ | ----------- | ---------- |
| 10–15M 步平均回报 | 20.65        | -10.28      | -5.08      |
| Episode 长度   | 19.70s       | 12.70s      | 1.48s      |
| XY 速度误差      | 0.143        | 0.629       | 0.889      |
| Yaw 速度误差     | 0.422        | 3.594       | 3.557      |
| 速度达标指标       | 98.94%       | 0%          | 0%         |

README.md

更值得注意的是，训练到 120–125M 步时：

- PG-only 的回报约为 3.33；
- PG+pMF 的回报约为 3.15；
- 两组都能维持约 20 秒，但 yaw 跟踪误差仍约为 3.07 rad/s；
- 加入 pMF Loss 后，策略更新计算耗时增加到约 3.87 倍，但没有观察到任务性能收益。

  README.md



这至少提示两个问题：

问题 A：当前 Joint Policy 的策略优化效率不足。

连不加 pMF Loss 的 PG-only 都远落后于 Gaussian PPO，说明不能简单地把困难归结为 MeanFlow Loss。

问题 B：当前 pMF Loss 没有改善任务表现。

这并不证明 pMF 没有意义，但说明目前的训练目标之间可能没有形成有效协同。

还有一个重要限制：两组算法的学习率、熵正则、网络规模、rollout 长度、探索方式等并不相同，因此暂时不能把性能差距直接归因于算法本身。

README.md



所以，我建议尝试 PMD，但目的不是给模型增加一个正则项，而是重新设计 Flow Policy 的改进和拟合过程。

## 二、为什么 PMD 可能比你现在的 PPO 更新更合适？

你目前的 Joint Policy 可以理解为：

\\[ z\sim\mathcal N(0,I) \\]

\\[ \mu\_\theta=F\_\theta(s,z) \\]

\\[ a\sim\mathcal N(\mu\_\theta,\sigma^2 I) \\]

然后利用联合潜变量概率比执行 PPO 更新。

问题是：PPO 优化的是执行动作的概率，而 pMF Loss 要求网络学习目标动作分布的生成结构。

两者不是数学上矛盾，但它们可能给出不同的参数更新方向。特别是 pMF Loss 如果拟合旧策略产生的动作，就有可能抵消一部分 PPO 的策略更新。

而 PMD 的思路是：

\\[ \boxed{ \pi_k \xrightarrow{\text{Policy Improvement}} \pi\_{\mathrm{MD}} \xrightarrow{\text{pMF Projection}} \pi\_{k+1} } \\]

把策略改进和生成式分布拟合分开。

这不是纯粹的猜想。One-Step Flow Policy Mirror Descent（FPMD） 已经提出 MeanFlow 版本的 PMD；FMER 也把 Advantage-weighted Flow Matching 作为 PMD 的代理优化，并使用有效样本量调节温度。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



因此可以把它们作为你下一步的算法依据，但不能把简单结合 PMD 当作全新贡献。

## 三、我建议设计一个 pMF-PMD 版本

这里有一个与你当前 Joint Policy 特别相关的细节。

你实际执行的动作并不是 \\(F\_\theta(s,z)\\)，而是：

\\[ a=F\_\theta(s,z)+\sigma\xi \\]

所以我建议先让 PMD 优化 pMF 的生成动作分布，再保留末端 Gaussian 作为探索机制。

记：

\\[ \mu=F\_\theta(s,z),\qquad \rho\_\theta(\mu\mid s)=F\_\theta(s,\cdot)\_\\#p_0 \\]

由于执行时还有探索噪声，定义平滑后的 Critic：

\\[ \bar Q_k(s,\mu) = \mathbb E\_{\xi} \left[ Q^{\pi_k}(s,\mu+\sigma\xi) \right] \\]

在生成动作分布 \\(\rho_k\\) 上做 PMD：

\\[ \boxed{ \rho\_{\mathrm{MD}}(\mu\mid s) \propto \rho_k(\mu\mid s) \exp\left( \frac{\bar Q_k(s,\mu)}{\eta} \right) } \tag{1} \\]

再把目标分布交给 pMF：

\\[ \boxed{ \mathcal L\_{\mathrm{PMD-pMF}} = \mathbb E\_{s,\mu\sim\rho_k} \left[ w(s,\mu)\\, \mathcal L\_{\mathrm{pMF}}(\theta;s,\mu) \right] } \tag{2} \\]

其中：

\\[ w(s,\mu)= \frac{e^{\bar Q_k(s,\mu)/\eta}} {\mathbb E\_{\mu'\sim\rho_k}e^{\bar Q_k(s,\mu')/\eta}} \\]

这个方案的好处是保留了 pMF 的 Flow Loss，同时让 Flow 学习的目标随策略改进而变化。

注意，这里是在生成动作均值分布上进行 PMD，不完全等于最终执行动作分布上的标准 PMD。它适合你目前的结构，但需要作为独立的算法变体进行验证。

### 具体实现流程

旧 pMF Actor

对每个状态采样 \\(M\\) 个候选动作均值 \\(\mu_1,\ldots,\mu_M\\)

Critic Evaluation

对候选动作估计 \\(\bar Q(s,\mu_i)\\)

PMD Weighting

\\(w_i=\operatorname{softmax}\_i(\bar Q_i/\eta)\\)

Weighted pMF Loss

使用独立的新 Flow 噪声，将生成器拟合到加权后的动作分布

这里有三个实现要点：

第一，每个状态最好采样多个候选动作，建议原型从 \\(M=8\\) 开始，而不是直接对 rollout 中每个状态唯一的动作做全局 softmax。

第二，执行 pMF Loss 时，应重新采样独立的 Flow 起点噪声。不要简单复用产生候选动作的原 latent，否则可能只是学习旧生成映射，达不到重新分配动作概率质量的目的。

第三，不要立即把现有的 PPO Loss、PMD Loss、pMF Loss 三者全部加在一起。先测试纯 PMD Improvement + pMF Projection，才能判断 PMD 是否有效。

## 四、但我会先检查一个更严重的问题：探索噪声

你的实验报告里，有一项值得特别关注：

| 10–15M 步            | Gaussian PPO | Joint Policy |
| ------------------- | ------------ | ------------ |
| Gaussian sigma 均值   | 0.7155       | 0.9852       |
| Entropy coefficient | 0.01         | 0            |

README.md

README.md

这两种 sigma 不能直接比较：Gaussian PPO 的 sigma 描述完整条件策略的探索，而 Joint Policy 的 sigma 只包含末端 Gaussian，没有计入 latent map 引入的动作变化。

对于 Joint Policy：

\\[ a=F\_\theta(s,z)+\sigma\xi \\]

固定状态下，如果两种噪声独立：

\\[ \boxed{ \operatorname{Cov}(a\mid s) = \operatorname{Cov}\_z[F\_\theta(s,z)] + \sigma^2 I } \\]

也就是说，Joint Policy 的总动作方差可能比日志中的 0.9852 所暗示的更大。

在 H1 这样的接触动力学任务中，我会高度怀疑过大的早期探索导致步态难以稳定形成，从而拖慢策略学习。这是待验证的机制，不是现有日志已经证明的原因。

我建议立即做一个诊断：固定同一批状态，分别对 Gaussian PPO 和 Joint Policy 采样 256 次动作，比较完整动作协方差、各维标准差以及控制动作的变化幅度。

如果 Joint Policy 的动作分布明显更分散，那么现在直接加 PMD 可能只是掩盖了基本探索问题。

另外，你报告中的更新设置还存在自适应学习率、value clipping、EMA 等差异，都应在受控消融中处理。

## 五、下一轮实验我建议这样安排

| 实验                       | 改动                              | 目的                     |
| ------------------------ | ------------------------------- | ---------------------- |
| A：Joint PG-only 修正版      | 对齐 PPO 的优化设置，检查探索噪声             | 确认 Joint Policy 本身能否学好 |
| B：pMF-PMD                | 用式 (2) 替代当前的联合 PPO Actor Update | 判断 PMD 是否改善学习          |
| C：pMF-PMD + Trust Region | 加入生成映射的 KL 约束                   | 判断更新稳定性是否提升            |
| D：pMF-PPO + 原 pMF Loss   | 保留当前方案作为对照                      | 隔离 PMD 的实际贡献           |

四组尽量使用相同的环境步预算、seed 和 Actor/Critic 容量，并做至少多个独立 seed 的复核。第一轮先观察 0–15M 环境步是否出现明显改善，而不是重新训练到 125M。

对于 B，我建议采用一个额外的保护措施：限制 PMD 权重的集中程度。可以根据 Effective Sample Size 调整温度 \\(\eta\\)：

\\[ \mathrm{ESS} = \frac{1}{\sum\_{i=1}^{M}w_i^2} \\]

如果 ESS 太低，说明更新依赖极少数候选动作，应增大 \\(\eta\\) 或降低更新幅度。前提是 Critic 足够可靠，否则指数加权会放大错误的 Q 估计。

## 六、一个快速检验：先让 pMF 模仿已经训练好的 PPO

在你现有实验里，这可能比直接引入 PMD 更快定位问题。

你已经有性能不错的 Gaussian PPO。

可以冻结它，用它在 Flat-H1 状态下生成动作，利用普通 pMF Loss 训练一个单步 pMF Actor。

然后在相同状态分布和独立评估条件下，比较：

- pMF 是否能够拟合 PPO 的动作分布；
- 模仿出来的 pMF Policy 是否也能保持站立和速度跟踪；
- 一步生成与多步 Flow 积分是否一致。

如果模仿效果都很差，问题更可能位于 pMF 的训练构造、网络表达或动作分布处理，而不在 PPO 与 PMD 的选择。

如果模仿很好，但从零开始在线训练很差，才更有理由聚焦策略改进方法。

这属于诊断性 Teacher Distillation 实验，不应直接拿使用 PPO 预训练的 pMF 与从头训练的 PPO 做公平样本效率比较。

## 七、我的最终判断

我建议你继续这条研究线，但改变推进优先级：

当前最高优先级：找到 Joint PG-only 明显落后的原因。

如果这个基础没有解决，PMD 很可能只是引入更多可调参数和 Critic 估计误差。

下一条算法分支：PMD + pMF Projection，而不是 PPO + PMD + pMF 三重 Loss。

这样可以直接检验 Flow Matching 是否能够实现有效的策略改进。

论文贡献：研究投影误差与策略改进，而不是重复提出 Advantage-weighted MeanFlow。

因为 FPMD 已经有 MeanFlow PMD 版本，后续需要证明你的方案在 pMF 参数化、优化稳定性或策略改进界方面具有实质区别。

最核心的一句话：

你现在不一定需要一个更复杂的 Loss，而是需要先证明两件事：

\\[ \boxed{ \begin{aligned} &\text{1. pMF 能正确学习目标策略分布}\\\ &\text{2. PMD 能持续产生更好的目标策略} \end{aligned} } \\]

只要这两件事成立，你的 pMF-PMD 才具备一条清晰的学习链路。与当前把 Policy Gradient 和 pMF Loss 同时作用在同一个 Actor 上相比，这会是一个更容易分析和验证的研究方案。
