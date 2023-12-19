## REINFORCE：初版策略梯度

**REINFORCE**算法在策略的参数空间中直观地通过梯度上升的方法逐步提高策略 $\pi_\theta$ 的性能。

$$ \nabla_\theta J(\pi_\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^\mathrm{T}R_t\nabla_\theta\sum_{t^{\prime}=0}^t\log\pi_\theta(A_{t^{\prime}}|S_{t^{\prime}})\right]=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t^{\prime}=0}^\mathrm{T}\nabla_\theta\log\pi_\theta(A_{t^{\prime}}|S_{t^{\prime}})\sum_{t=t^{\prime}}^\mathrm{T}R_t\right]\tag{1} $$

注意：上述式子中 $\sum_{t=i}^{\mathrm{T}}R_t$ 可以看成是智能体在状态 $S_i$ 处选择动作 $A_i$，并在之后执行当前策略的情况下，从第 $i$ 步开始获得的累计奖励。事实上，$\sum_{t=i}^{\mathrm{T}}R_t$ 也可以看成 $Q_i(A_i,S_i)$, 在第 $i$ 步状态$S_i$ 处采取动作 $A_i$, 并在之后执行当前策略的 $Q$ 值。

<font color=yellow>通过给不同的动作所对应的梯度根据它们的累计奖励赋予不同的权重，鼓励智能体选择那些累计奖励较高的动作 $A_{i}$。</font>

只要把上述式子中的 $T$ 替换成 $\infty$ 并赋予$R_t$ 以 $\gamma^t$ 的权重，扩展到折扣因子为 $\gamma$ 的无限范围。

$$\nabla J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t'=0}^\infty\nabla_\theta\log\pi_\theta(A_{t'}|S_{t'})\gamma^{t'}\sum_{t=t'}^\infty\gamma^{t-t'}R_t\right]\tag{2}$$

由于折扣因子给未来的奖励赋予了较低的权重，使用折扣因子还有助于减少估计梯度时的方差大的问题。实际使用中， $\gamma^{t^{\prime}}$ 经常被去掉，从而避免了过分强调轨迹早期状态的问题。

- 优点
  - 简单直观
- 缺点
  - 对梯度的估计有较大的方差<sup><a href="#ref1">1</a></sup>。

> 1. <p name = "ref1">对于一个长度为 L 的轨迹，奖励 $R_t$ 的随机性可能对 L 呈指数级增长。</p>