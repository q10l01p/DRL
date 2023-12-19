# DDPG

DDPG的关键组成部分包括：

1. 回放缓冲区（Replay Buffer）

2. 演员-评论家神经网络（Actor-Critic Neural Network）

3. 探索噪声（Exploration Noise）

4. 目标网络（Target Network）

5. 用于目标网络的软目标更新（Soft Target Updates for Target Network）

<div align=center>
<img width="550" src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*Ep5xRjinEnakXULKM4ENWw.png"/>
</div>
<div align=center></div>

## Replay Buffer

DDPG使用回放缓冲区来存储通过探索环境采样的转换和奖励$(S_t, A_t, Rₜ, S_{t+1})$。回放缓冲区在DDPG的学习加速和稳定性方面发挥着关键作用，具体表现在：

1. **最小化样本之间的相关性**：通过将过去的经验存储在回放缓冲区中，减小了样本之间的相关性，使得代理能够从多样化的经验中学习。
2. **实现异策略学习**：允许代理从回放缓冲区中采样转换，而不是根据当前策略采样转换。
3. **提高样本效率**：将过去的经验存储在回放缓冲区中，使得代理能够多次从多样化的经验中学习。

```python
class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        """
        经验回放缓冲区类

        参数:
        - state_dim: 状态空间维度
        - action_dim: 动作空间维度
        - max_size: 缓冲区的最大容量
        - dvc: 存储设备（通常为 GPU）

        说明:
        1. 初始化经验回放缓冲区的各项属性。
        2. 使用零张量初始化存储状态、动作、奖励、下一个状态和终止标志的张量。
        """
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)  # 存储状态的张量
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)  # 存储动作的张量
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)  # 存储奖励的张量
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)  # 存储下一个状态的张量
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)  # 存储终止标志的张量

    def add(self, s, a, r, s_next, dw):
        """
        向经验回放缓冲区添加新的经验样本

        参数:
        - s: 当前状态
        - a: 执行的动作
        - r: 获得的奖励
        - s_next: 下一个状态
        - dw: 终止标志

        说明:
        1. 将输入的经验样本添加到经验回放缓冲区。
        2. 使用循环索引实现循环缓冲。
        """
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)  # 将当前状态添加到缓冲区
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)  # 将执行的动作添加到缓冲区
        self.r[self.ptr] = r  # 将获得的奖励添加到缓冲区
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)  # 将下一个状态添加到缓冲区
        self.dw[self.ptr] = dw  # 将终止标志添加到缓冲区

        self.ptr = (self.ptr + 1) % self.max_size  # 更新循环索引
        self.size = min(self.size + 1, self.max_size)  # 更新缓冲区大小

    def sample(self, batch_size):
        """
        从经验回放缓冲区中随机采样批次数据

        参数:
        - batch_size: 批次大小

        返回:
        - s: 当前状态的张量
        - a: 执行的动作的张量
        - r: 获得的奖励的张量
        - s_next: 下一个状态的张量
        - dw: 终止标志的张量

        说明:
        1. 随机选择一批样本进行返回，以用于训练。
        """
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))  # 随机生成索引
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]  # 返回随机批次的数据
```

## Actor-Critic神经网络

这是一个用于连续控制任务的Actor-Critic强化学习算法的PyTorch实现。

该代码定义了两个神经网络模型，一个是Actor，另一个是Critic。

- Actor模型的输入：环境状态
- Actor模型的输出：具有连续值的动作
- Critic模型的输入：环境状态和动作
- Critic模型的输出：当前状态-动作对的期望总奖励的Q值。

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        """
        Actor 神经网络，用于输出动作

        参数:
        - state_dim: 状态空间维度
        - action_dim: 动作空间维度
        - net_width: 神经网络宽度
        - maxaction: 动作的最大取值

        说明:
        1. 初始化网络结构，包括三个全连接层。
        2. 输出层通过 tanh 激活函数并乘以 maxaction，将动作范围映射到 [-maxaction, maxaction]。
        """
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)  # 第一个全连接层
        self.l2 = nn.Linear(net_width, 300)  # 第二个全连接层
        self.l3 = nn.Linear(300, action_dim)  # 输出层

        self.maxaction = maxaction

    def forward(self, state):
        """
        正向传播

        参数:
        - state: 输入的状态

        返回:
        - a: 输出的动作

        说明:
        1. 通过两个 ReLU 激活函数传播信号。
        2. 输出层通过 tanh 激活函数，并乘以 maxaction，映射到 [-maxaction, maxaction] 的范围。
        """
        a = torch.relu(self.l1(state))  # 第一个激活函数
        a = torch.relu(self.l2(a))  # 第二个激活函数
        a = torch.tanh(self.l3(a)) * self.maxaction  # 输出动作，并映射到 [-maxaction, maxaction]
        return a
```

```python
class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        """
        Q-Value Critic 神经网络，用于计算 Q 值

        参数:
        - state_dim: 状态空间维度
        - action_dim: 动作空间维度
        - net_width: 神经网络宽度

        说明:
        1. 初始化网络结构，包括三个全连接层。
        2. 输入层将状态和动作连接在一起。
        """
        super(Q_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, net_width)  # 输入层
        self.l2 = nn.Linear(net_width, 300)  # 第二个全连接层
        self.l3 = nn.Linear(300, 1)  # 输出层

    def forward(self, state, action):
        """
        正向传播

        参数:
        - state: 输入的状态
        - action: 输入的动作

        返回:
        - q: 计算得到的 Q 值

        说明:
        1. 将状态和动作连接在一起。
        2. 通过两个 ReLU 激活函数传播信号。
        3. 输出层计算 Q 值。
        """
        sa = torch.cat([state, action], 1)  # 将状态和动作连接在一起
        q = F.relu(self.l1(sa))  # 第一个激活函数
        q = F.relu(self.l2(q))  # 第二个激活函数
        q = self.l3(q)  # 输出 Q 值
        return q
```

## 探索噪声

在DDPG中，向演员选择的动作添加噪声是一种技术，用于促进探索并改善学习过程。

你可以选择使用**高斯噪声**或奥恩斯坦-乌伦贝克（**Ornstein-Uhlenbeck**）噪声。高斯噪声实现简单，而奥恩斯坦-乌伦贝克噪声生成具有时间相关性的噪声，有助于代理更有效地探索动作空间。

**奥恩斯坦-乌伦贝克噪声**的波动更加**平滑**，比高斯噪声方法**更具有规律性**。

```python
import numpy as np
import random
import copy

class OU_Noise(object):
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        初始化 Ornstein-Uhlenbeck 噪声生成器

        参数:
        - size: 噪声维度
        - seed: 随机种子
        - mu: 噪声均值，默认为0
        - theta: Ornstein-Uhlenbeck 过程的回复速度
        - sigma: 控制噪声的强度

        说明:
        1. mu 为噪声的均值。
        2. theta 控制 Ornstein-Uhlenbeck 过程的回复速度。
        3. sigma 控制噪声的强度。
        4. seed 用于生成随机数的种子。
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """重置内部状态（噪声）为均值（mu）。"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        更新内部状态并返回作为噪声样本。
        该方法使用当前噪声状态并生成下一个样本。
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state
```

## DDPG

在DDPG（深度确定性策略梯度）中，采用了**两组演员-评论家神经网络**进行函数逼近。

> 在DDPG中，目标网络是演员-评论家网络的对应物。目标网络与演员-评论家网络具有**相同的结构和参数化**。

在训练过程中，代理使用其演员-评论家网络与环境进行交互，并将经验元组$(S_t, A_t, R_t, S_{t+!})$存储在回放缓冲区中。代理然后从回放缓冲区中采样，并使用数据来更新演员-评论家网络。然而，DDPG算法不是通过直接复制演员-评论家网络的权重来更新目标网络的权重，而是通过一种称为**软目标更新**的过程缓慢地更新**目标网络的权重**。

> 软目标更新是从演员-评论家网络传输到目标网络的一小部分权重，称为目标更新率（$\tau$）。

软目标更新的实现如下：

$$\begin{aligned}\theta^{Q^{\prime}}&\leftarrow\tau\theta^Q+(1-\tau)\theta^{Q^{\prime}}\\\theta^{\mu^{\prime}}&\leftarrow\tau\theta^\mu+(1-\tau)\theta^{\mu^{\prime}}\end{aligned}$$

软目标技术极大地提高了学习的稳定性。

```python
class DDPG_agent():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)

    def select_action(self, state, deterministic):
        """
        选择动作

        参数:
        - state: 当前状态
        - deterministic: 是否使用确定性策略选择动作

        返回:
        - 动作

        说明:
        1. 将状态从 [x, x, ..., x] 转换为 [[x, x, ..., x]]。
        2. 通过 Actor 网络获取动作。
        3. 如果是确定性策略，则直接返回动作。
        4. 如果是非确定性策略，则添加噪声并返回，确保在动作范围内。
        """
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # 1. 将状态转换为网络输入格式
            a = self.actor(state).cpu().numpy()[0]  # 2. 获取动作

            if deterministic:
                return a  # 3. 返回动作（确定性策略）

            else:
                noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)  # 4. 生成噪声
                return (a + noise).clip(-self.max_action, self.max_action)  # 5. 添加噪声并确保在动作范围内
        
    def train(self):
        """
        训练方法

        说明:
        1. 从经验回放缓冲区中采样一批数据，包括当前状态(s)、动作(a)、奖励(r)、下一个状态(s_next)和终止标志(dw)。
        2. 使用目标策略网络(actor_target)预测下一个状态的动作(target_a_next)。
        3. 使用目标评论网络(q_critic_target)计算目标Q值(target_Q)。
        4. 计算目标Q值，考虑折扣因子(gamma)和终止标志(dw)。
        5. 计算当前评论网络(q_critic)的Q值(current_Q)。
        6. 计算Q值损失(q_loss)并执行评论网络优化。
        7. 计算动作损失(a_loss)并执行策略网络优化。
        8. 更新目标评论网络和目标策略网络参数，使用软更新策略。

        注意:
        - 该方法执行了一次深度强化学习的训练迭代。
        """
        with torch.no_grad():
            s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)  # 1. 采样经验数据
            target_a_next = self.actor_target(s_next)  # 2. 预测下一个状态的动作
            target_Q = self.q_critic_target(s_next, target_a_next)  # 3. 计算目标Q值
            target_Q = r + (~dw) * self.gamma * target_Q  # 4. 计算目标Q值，考虑折扣因子和终止标志

        current_Q = self.q_critic(s, a)  # 5. 计算当前评论网络的Q值

        q_loss = F.mse_loss(current_Q, target_Q)  # 6. 计算Q值损失

        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()  # 6. 执行评论网络优化

        a_loss = -self.q_critic(s, self.actor(s)).mean()  # 7. 计算动作损失
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()  # 7. 执行策略网络优化

        with torch.no_grad():
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)  # 8. 更新目标网络参数，使用软更新策略

    def save(self,EnvName, timestep):
            torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
            torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

    def load(self,EnvName, timestep):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))
```

## 结论

DDPG深度强化学习算法是一种无模型的离策略演员-评论家算法，受到Deep Q-Network（DQN）算法的启发。它结合了策略梯度方法和Q学习的优点，以学习连续动作空间的确定性策略。

与DQN类似，DDPG使用回放缓冲区存储过去的经验和目标网络，用于训练网络，从而提高训练过程的稳定性。软目标更新目标网络的权重。

DDPG使用探索噪声来探索环境并收集样本。

通过使用演员-评论家网络，DDPG代理使用策略梯度方法学习优化的策略。

DDPG算法需要仔细调整超参数以获得最佳性能。这些超参数包括学习率、批大小、目标网络更新率和探索噪声参数。超参数的微小变化可能对算法的性能产生显著影响。

