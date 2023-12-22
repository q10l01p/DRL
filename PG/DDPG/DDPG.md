# DDPG
<div align=center>
<img width="550" src="./png/DDPG.svg "/>
</div>
<div align=center></div>
## 连续控制问题

在强化学习中，一个连续控制问题是指代理需要从一个连续动作空间中选择动作，这比离散动作空间复杂得多。

> 在**离散**动作空间中，智能体可选择的动作数量**有限**；然而，在**连续**控制问题的情况下，智能体必须从更大且通常是**无限**数量的动作中进行选择，使得最佳动作选择成为一个复杂的问题。

### 可行算法

解决连续控制问题的强化学习算法通常基于**策略梯度**，其中代理学习一个直接将状态映射到动作的策略，例如：

- Deep Deterministic Policy Gradient(DDPG)

- Proximal Policy Gradient(PPO)

- Trust Region Policy Optimization(TRPO)

- Soft Actor-Critic(SAC)

> 深度确定性策略梯度（DDPG）是一种**无模型**、离线深度强化学习算法，受到**深度 Q 网络**的启发，基于使用策略梯度的演员-评论家（Actor-Critic）结构。

## 术语介绍

### 确定性策略

确定性表示在确定性系统的**输出**中**没有随机性或变异性**，这与随机策略形成对比。

> 确定性策略意味着在强化学习环境中，对于**给定的状态**，系统将始终**生成相同的动作**。相反，**随机策略**生成动作的**概率分布**，代理根据这个分布**随机选择**动作。

**深度确定性策略梯度**（DDPG）之所以被称为“确定性”，是因为它学习了一个**将状态映射到动作**的**确定性策略**。因此，给定一个状态，DDPG算法将始终产生相同的动作。

### 无模型强化学习算法

一个基于模型或无模型的强化学习算法的确定性取决于代理**如何通过与环境的交互**进行学习。

#### **基于模型的强化学习**

> 代理利用一个**模型**来**预测状态转换和奖励的结果**。通过这个模型，代理可以通过确定在不同状态和可能的动作下哪些动作会产生最高奖励来学习最优策略。然而，基于模型的强化学习需要一个**精确而完整的环境模型**。

基于模型的强化学习的一个例子是代理试图学习象棋、围棋或扑克等游戏。该算法可以**学习游戏规则**，模拟可能的未来走法，并相应地**规划自己的策略**。

#### **无模型强化学习算法**

> 代理通过**探索环境**并通过**反复试验**来**积累经验**进行学习。代理尝试各种动作以了解哪些动作产生更好的结果，并相应地调整其策略。与基于模型的强化学习不同，无模型的强化学习**不需要环境的真实模型**。

基于无模型的强化学习的一个例子机器人控制，其中动作空间是连续且广泛的，使得建立一个模型以找到每个可能状态的最优动作变得不可行。

### off-policy

off-policy方法涉及两种不同的策略：行为策略和目标策略。

- **行为策略**（$\pi$）用于**探索环境**并**收集样本**。行为策略通过根据某种探索策略选择动作来生成代理的行为。
- **目标策略**（$v$）是通过最大化代理行为的期望累积奖励来学习和优化的。

这种方法允许代理从由**行为策略生成的过去经验中学习**，并**利用这些知识来改进目标策略**。

<div align=center>
<img width="550" src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*ekXMKiCIwouMZ6NF.png "/>
</div>
<div align=center></div>

### DQN

深度 Q 网络（Deep Q-Network，DQN）使用**深度神经网络**来**学习**离散动作空间的 **Q 函数**近似。

$$\text{DQN(Deep Q-Network)= Q 学习 + 人工神经网络}$$

DQN通过使用Q学习的一种新颖变体和两个关键思想来解决强化学习中深度神经网络的不稳定性。

- **经验回放**（Experience Replay）：网络采用经验回放缓冲区中的样本进行off-policy训练，以减小样本之间的相关性，使代理能够重新访问并学习过去的经验。

- **目标网络**（Target Network）：通过使用目标Q网络进行训练，这是Q网络的一个副本，用于近似Q函数，从而在时间差分更新时提供一致的目标。

  <div align=center>
  <img width="550" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*kiyVYCNry7I6GDdz8wuK7Q.png "/>
  </div>
  <div align=center></div>

DQN假设动作空间是**离散**的。对于连续动作空间，DQN代理需要从非常庞大且无限的动作中选择一个动作，这使得无法为动作空间中的每个可能动作表示Q函数变得不切实际。

在Q学习中找到**贪婪策略**需要在每个时间步对动作进行优化；但是，对于大型、无约束的函数逼近器和复杂的动作空间来说，这种优化速度过慢，不切实际。

$$\begin{align}Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha{\left[R_{t+1}+\gamma\max_aQ(S_{t+1},a)-Q(S_t,A_t)\right]}\tag{1}\end{align}$$

Q学习：代理从状态$S_t$开始，执行动作$A_t$并获得奖励$R_{t+1}$，然后在状态$S_{t+1}$中选择具有最大可能奖励的$A_{t+1}$，并更新状态$S_t$中动作$A_t$的$Q$值。

### Actor-Critic 

> **演员-评论家**强化学习旨在通过两个网络（演员和评论家）为代理在环境中找到最优策略。这是一种基于**价值**和基于策略的方法的组合，其中**演员**使用**策略梯度**控制代理的**行为**，而**评论家**基于值函数评估代理所采取的动作的**好坏**
>
> 。

- 演员（Actor）：演员通过**探索**环境学习最优**策略**。

- 评论家（Critic）：评论家**评估**演员采取的每个动作的价值，以确定该动作**是否会导致更好的奖励**，指导演员采取最佳行动。

演员利用评论家的反馈调整其策略，做出更明智的决策，从而提高整体性能。

<div align=center>
<img width="550" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*sWqG-2hbQZ86SW4ftH3BDA.png "/>
</div>
<div align=center></div>

## DDPG

DDPG（深度确定性策略梯度）是一种强化学习算法，基于DQN中的技术。具体而言，DDPG利用了DQN中的两种技术。

- **Replay buffer**
- **Target network**

除了这些技术之外，DDPG还使用了**两组演员-评论家**神经网络进行函数近似。这两组都包括一个演员网络和一个评论家网络，它们具有**相同的结构和参数化**。

<div align=center>
<img width="550" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*xgI6Y5V-MOwi4A9j6VP-1Q.png "/>
</div>
<div align=center></div>

### **Soft Target Updates** 

> 在DDPG中，采用**软目标更新**，以**缓慢更新目标网络的权重**，而**不是直接复制**演员-评论家网络的权重。

软目标技术极大地提高了学习的稳定性，软目标更新的实现方式如下：

$$\begin{aligned}\theta^{Q^{\prime}}&\leftarrow\tau\theta^Q+(1-\tau)\theta^{Q^{\prime}}\\\theta^{\mu^{\prime}}&\leftarrow\tau\theta^\mu+(1-\tau)\theta^{\mu^{\prime}}\end{aligned}$$

在每个时间步，从演员-评论家网络中转移一小部分权重到目标网络。这一部分的确定由一个称为**目标更新率**（$\tau$）的超参数决定。

### **Replay Buffer**

DDPG使用回放缓冲区来存储通过探索环境采样的转换和奖励$(S_t，A_t，R_t，S_{t+1})$。

- **最小化样本之间的相关性**，允许代理重新访问并从过去的经验中学习。

- **使得算法可以是off-policy的**，因为代理可以从缓冲区中采样，而不是每次都根据当前策略采样一个新的轨迹。

- 回放缓冲区通过存储过去的经验使得DDPG算法具有**高效的样本利用性**，允许算法多次从各种经验中学习，从而减少了总体所需的样本数量。

为确保回放缓冲区可管理，当其达到最大容量时，最老的样本会被丢弃。这允许缓冲区保持固定大小，同时保留代表性的经验集。

### **Actor-Critic network**

DDPG算法涉及一个**演员**学习一个策略$π_θ$，以及一个**评论家**使用Q函数**评估**演员网络的**动作**。DDPG有一个**目标网络**的对应物，分别是$π_θ’$和$Q’$。DDPG通过向演员选择的**动作添加探索噪声**，以解决在连续动作空间中进行探索的挑战。

1.  $π(S;θ_µ)$：**演员**，具有参数 $θ$，将来自回放缓冲区的观测 $S_t$ 作为**输入**，并返回最大化长期奖励$R_t$ 的相应动作$A_t$，从而学习策略$µ$。
2. Q(S, A)：**评论家**，具有参数 $θ'$，从回放缓冲区中获取观测 $S_t$ 和动作$A_t$（即演员神经网络在添加噪声后的输出）作为**输入**，并输出相应的期望 $Q(s, a)$，使用贝尔曼最优性方程的状态-动作值函数。
3.  $πₜ(Sₜ₊₁;θ^{µ'})$：**目标演员**，将**下一个状态**（$S_{t+1}$）作为输入传递给**目标演员**。代理定期使用最新的演员参数值更新目标演员参数 $θ_{µ'}$。目标网络提高了优化的稳定性。

$$\theta^{\mu^{\prime}}\leftarrow\tau\theta^\mu+(1-\tau)\theta^{\mu^{\prime}}$$

4. $Q’(S_{t+1}, A_{t+1})$：**目标评论家**，代理定期使用**最新的评论家参数值**更新**目标评论家参数**$ θ^{Q’}$。

$$\theta^{Q^{\prime}}\leftarrow\tau\theta^Q+(1-\tau)\theta^{Q^{\prime}}$$

Target Network输出：

$$\begin{aligned}y_i=r_i+\gamma Q'(s_{i+1},\mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})\end{aligned}$$

更新评论家网络，通过最小化如下计算的损失：



### **Exploration Noise**

一探索策略 $µ'$ ，通过将从**Ornstein-Uhlenbeck过程**中采样的噪声（$N$）添加到其中。

$$\mu'(s_t)=\mu(s_t|\theta_t^\mu)+\mathcal{N}$$

向演员选择的动作$A_t$添加噪声，用于**鼓励探索**并**改善学习**过程。

$$a_t=\mu(s_t|\theta^\mu)+\mathcal{N}_t$$

通过添加探索噪声 $N$，**鼓励代理采取它可能不会选择的动作**，从而对状态-动作空间进行更全面的探索，潜在地提高性能。

$$L=\frac1N\sum_i(y_i-Q(s_i,a_i|\theta^Q))^2$$

在每个时间步，Actor 和 Critic 都会通过从缓冲区中**均匀采样小批量**来更新。

## 伪代码

<div align=center>
<img width="550" src="https://spinningup.openai.com/en/latest/_images/math/5811066e89799e65be299ec407846103fcf1f746.svg "/>
</div>
<div align=center></div>

## 结论

DDPG是一种无模型算法，用于在大型离散或连续动作空间中找到最优的确定性策略。它采用离策略方法，具有较高的样本效率，适用于样本获取成本较高的环境。该算法采用演员-评论家架构，结合了基于值和基于策略方法的优点。

DDPG对于嘈杂或不完整的观测相对较为鲁棒，可以从部分观测中学习有效的策略。然而，该算法收敛较慢，需要仔细调整超参数，包括学习率和批大小，以获得最佳性能。

## 参考

1. [Unlock the secrets of DDPG in Reinforcement Learning ](http://webcache.googleusercontent.com/search?q=cache:https://medium.com/@arshren/unlock-the-secrets-of-ddpg-in-reinforcement-learning-61f0db5035bb&strip=0&vwsrc=1&referer=medium-parser)
2. [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf)
3. [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#id1)
