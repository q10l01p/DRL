# PG

## REINFORCE

直接用神经网络 $magic(s)=a$ 不行吗？这就是PG的基本思想。

PG用的是MC的G值来更新网络。也就是说，PG会让智能体一直走到最后。然后通过回溯计算G值。

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-9f7bbfc8d7cf4e7424f9dcd9d5ae4be1_720w.webp "/>
</div>
<div align=center></div>

于是得到S - A - G的数据。这里的G就是对于状态S，选择了A的评分。

- 如果G值**正数**，那么表明选择A是**正确**的，我们希望神经网络输出A的**概率增加**。(鼓励)
- 如果G是**负数**，那么证明这个选择**不正确**，我们希望神经网络输出A**概率减少**。(惩罚)
- 而G值的大小，就相当于**鼓励和惩罚的力度**了。

我们分别以ABC三途路径作为例子：

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-b0252a3e09676a946e38ff3c8c3a4431_720w.webp "/>
</div>
<div align=center></div>

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-5a6cf01c21f4b27c2c8a3828a4b45bf4_720w.webp "/>
</div>
<div align=center></div>

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-b097923d75bb7559bb7d2efd14c16210_720w.webp "/>
</div>
<div align=center></div>

为此，我们可以用**带权重的梯度**。

[如何理解策略梯度（Policy Gradient）算法](./REINFORCE/Policy_Gradient.md)

## Actor-Critic

MC的效率是相对比较低的，因为需要一直走到最终状态。所以我们希望**用TD代替MC**。

所以我们希望用TD代替MC。那么我们可不可以把PG和DQN结合呢？

注意：这里是一个大坑。个人更倾向于把AC理解成PG的TD版本，而不是PG+DQN。

1. Critic网络负责估算Q值
2. Actor网络负责估算策略

注意，Q值都是正数，容易掉进“**正数陷阱**”。

假设我们用Critic网络，预估到S状态下三个动作A1，A2，A3的Q值分别为1,2,10。

但在开始的时候，我们采用平均策略，于是随机到A1。于是我们用策略梯度的带权重方法更新策略，这里的权重就是Q值。

于是策略会更倾向于选择A1，意味着更大概率选择A1。结果A1的概率就持续升高...

那要怎么办？我们把Q值弄成有正有负就可以了。一堆数减去他们的平均值一定有正有负吧！Q减去Q的期望值，也就是V值，就可以得到有正有负的Q了。

也就是说Actor用**Q(s,a)-V(s)**去更新。但我们之前也说过Q和V都要估算太麻烦了。能不能只统一成V呢？

<div align=center>
<img width="800" src="https://pic4.zhimg.com/80/v2-bb84b957eee9fcd821a78e5631d5ac57_720w.webp"/>
</div>
<div align=center></div>

Q(s,a)用 $\gamma * V(s^{\prime}) + r$ 来代替，于是整理后就可以得到:

$$\text{TD-error:}\quad\gamma * V(s^{\prime}) + r - V(s)$$

这个和之前DQN的更新公式非常像:

- DQN的更新用了Q
- TD-error用的是V

Critic是用来预估V值，而不是原来讨论的Q值。那么，这个TD-error是用来更新Critic的loss了！

<div align=center>
<img width="800" src="https://pic1.zhimg.com/80/v2-06c9787f9cd9a71d92ce0bbeb871af60_720w.webp"/>
</div>
<div align=center></div>

[理解Actor-Critic的关键是什么？](./AC/AC.md)

## PPO

在强化学习中，数据来自智能体和环境互动。所以，数据都弥足珍贵，我们希望尽量能够利用好每一份数据。

但AC是一个在线策略的算法，也就是行为策略跟目标策略并不是同一个策略。

- **行为策略**——不是当前策略，用于**产出数据**
- **目标策略**——会更新的策略，是**需要被优化的策略**

- 在线策略:如果两个策略是同一个策略，那么我们称为On Policy，在线策略。
- 离线策略:如果不是同一个策略，那么Off Policy，离线策略。

如果我们在智能体和环境进行互动时产生的数据打上一个标记。标记这是第几版本的策略产生的数据,例如 1， 2... 10

现在我们的智能体用的策略 10，需要更新到 11。如果算法只能用 10版本的产生的数据来更新，那么这个就是**在线策略**；如果算法允许用其他版本的数据来更新，那么就是**离线策略**。

所以，我们需要用到**重要性更新**的，就可以用上之前策略版本的数据了。

[如何直观理解PPO算法？](./PPO/PPO.md)
