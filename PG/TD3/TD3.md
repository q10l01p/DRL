# 双延时确定策略梯度（TD3）

由于存在高估等问题，DPG 实际运行的效果并不好。

## 高估问题的解决方案

解决方案——截断双 Q 学习（clipped double Q-learning）：这种方法使用<font color='blue'>两个价值网络</font>和<font color='blue'>一个策略网络</font>：

$$
q(s,a;w_1),\quad q(s,a;w_2),\quad \mu(s;\theta)
$$
三个神经网络各对应一个目标网络：

$$
q(s,a;w_{1}^-),\quad q(s,a;w_{2}^-),\quad \mu(s;\theta^-)
$$

用<font color='green'>目标策略网络</font>计算<font color='green'>动作</font>：

$$
a_{j+1}^{-} = \mu(s_{j+1};\theta^{-})\tag{1}
$$

然后用两个目标价值网络计算：

$$
\begin{align}
\hat{y}_{j,1}=r_j+\gamma\cdot q(s_{j+1},\hat{a}_{j+1}^-;w_1^-)\tag{2}
\\
\hat{y}_{j,2}=r_j+\gamma\cdot q(s_{j+1},\hat{a}_{j+1}^-;w_2^-)\tag{3}
\end{align}
$$

取较小值为TD目标：
$$
\hat{y}_j = \min\{\hat{y}_{j,1}, \hat{y}_{j,2}\}\tag{4}
$$

截断双 Q 学习中的六个神经网络的关系如图 所示

<div align=center>
<img width="1200" src="./png/TD3神经网络之间的关系.svg"/>
</div>
<div align=center>TD3六个神经网络的关系图</div>

## 其他改进方法

### 往动作中加噪声

截断双 Q 学习用目标策略网络计算动作：目标策略网络 $a_{j+1}^{-} = \mu(s_{j+1};\theta^{-})$。把这一步改成：

$$
a_{j} = \mu(s_{};\theta_{\text{now}}) + \xi\tag{5}
$$
公式中的 ξ 是个随机向量，表示噪声，它的每一个元素独立随机从<font color='blue'>截断正态分布</font>（clippednormal distribution）中抽取。把截断正态分布记作 $CN(0,\sigma^2,-c,c)$, 意思是均值为零，标准差为$\sigma$ 的正态分布，但是变量落在区间 $[-c,c]$ 之外的概率为零。

正态分布与截断正态分布的对比如图所示。使用截断正态分布，而非正态分布，是为了防止噪声$\xi$过大。使用截断，保证噪声大小不会超过 $-c$ 和 $c$。

<div align=center>
<img width="1200" src="./png/正态分布与截断正态分布.png"/>
</div>
<div align=center>正态分布和截断正态分布</div>

### 减小更新策略网络和目标网络的频率

Actor-critic 用价值网络来指导策略网络的更新。如果价值网络 $q$ 本身不可靠，那么用价值网络 $q$ 给动作打的分数是不准确的，无助于改进策略网络 $\mu$。在价值网络 $q$ 还很差的时候就急于更新 $\mu$，非但不能改进 $\mu$，反而会由于 $\mu$ 的变化导致 $q$ 的训练不稳定。

 实验表明，<font color='red'>应当让策略网络 $\mu$ 以及三个目标网络的更新慢于价值网络 $q$</font>。传统的actor-critic 的每一轮训练都对策略网络、价值网络、以及目标网络做一次更新。更好的方法是每一轮更新一次价值网络，但是<font color='red'>每隔 $k$ 轮更新一次策略网络和三个目标网络</font>。$k$ 是超参数，需要调。

# 训练流程

TD3 与 DPG 都属于<font color='blue'>异策略</font> (off-policy), 可以用任意的行为策略收集经验，事后做经验回放训练策略网络和价值网络。

收集经验的方式与原始的训练算法相同，用 $a_t=\mu(s_t;\theta)+\epsilon$ 与环境交互，把观测到的四元组 $(s_t,a_t,r_t,s_{t+1})$ 存入经验回放数组。

```python
if total_steps < cfg['random_steps']:
    a = env.action_space.sample()
else:
    a = agent.select_action(s, deterministic=False)

s_next, r, dw, tr, info = env.step(a) # 与环境交互
r = Reward_adapter(r, cfg['env_name'])  # 调整奖励
done = (dw or tr)  # 如果游戏结束（死亡或胜利），则done为True
# 存储当前的转移数据
agent.replay_buffer.add(s, a, r, s_next, dw)
```

```python
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
            noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)  # 4. 生成噪声
        	return (a + noise).clip(-self.max_action, self.max_action)  # 5. 添加噪声并确保在动作范围内
```

```python
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
```

初始的时候，策略网络和价值网络的参数都是随机的。这样初始化目标网络的参数：
$$
w_1^-\leftarrow w_1,\quad w_2^-\leftarrow w_2,\quad \theta^- \leftarrow \theta.
$$
```python
self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)  # 3. 初始化演员网络
self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)  # 初始化演员网络的优化器
elf.actor_target = copy.deepcopy(self.actor)  # 初始化演员网络的目标网络

self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)  # 4. 初始化双 Q 评论者网络
self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)  # 初始化双 Q 评论者网络的优化器
self.q_critic_target = copy.deepcopy(self.q_critic)  # 初始化双 Q 评论者网络的目标网络
```

训练策略网络和价值网络的时候，每次从数组中随机抽取一个四元组，记作$(s_j,a_j,r_j,s_{j+1})$。

```python
s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
```

```python
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
用下标 now 表示神经网络当前的参数，用下标 new 表示更新后的参数。然后执行下面的步骤，更新价值网络、策略网络、目标网络。

1. 让目标策略网络做预测:  $\hat{a}_{j+1}^- = \mu(s_{j+1}; \theta_{\text{now}}^-)+\xi$。其中向量$\xi$ 的每个元素都独立从截断正态分布 $\mathcal{CN}(0,\sigma^2,-c,c)$ 中抽取。
   ```python
   # 生成目标动作的噪声并应用裁剪。
   target_a_noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
   
   # 计算平滑目标动作
   smoothed_target_a = (self.actor_target(s_next) + target_a_noise).clamp(-self.max_action, self.max_action)
   ```
   
2. 让两个目标价值网络做预测：
   $$
   \hat{q}_{1,j+1}^- = q(s_{j+1},\hat{a}_{j+1}^- ;w_{1,\text{now}} ^ - ) \quad \text{和} \quad \hat{q}_{2,j+1}^- = q(s_{j+1},\hat{a}_{j+1}^- ;w_{2,\text{now}} ^ - )
   $$
   ```python
   # 计算目标 Q 值
   target_Q1, target_Q2 = self.q_critic_target(s_next, smoothed_target_a)
   ```
   ```python
   class Double_Q_Critic(nn.Module):
       def __init__(self, state_dim, action_dim, net_width):
           """
           初始化双 Q 评论者网络
   
           参数:
           - state_dim: 状态空间维度
           - action_dim: 动作空间维度
           - net_width: 神经网络的宽度
           """
           super(Double_Q_Critic, self).__init__()
   
           # 第一个 Q 网络
           self.l1 = nn.Linear(state_dim + action_dim, net_width)
           self.l2 = nn.Linear(net_width, net_width)
           self.l3 = nn.Linear(net_width, 1)
   
           # 第二个 Q 网络
           self.l4 = nn.Linear(state_dim + action_dim, net_width)
           self.l5 = nn.Linear(net_width, net_width)
           self.l6 = nn.Linear(net_width, 1)
   
       def forward(self, state, action):
           """
           前向传播函数
   
           参数:
           - state: 当前状态
           - action: 采取的动作
   
           返回:
           - q1: 第一个 Q 值
           - q2: 第二个 Q 值
           """
           sa = torch.cat([state, action], 1)
   
           # 计算第一个 Q 值
           q1 = F.relu(self.l1(sa))
           q1 = F.relu(self.l2(q1))
           q1 = self.l3(q1)
   
           # 计算第二个 Q 值
           q2 = F.relu(self.l4(sa))
           q2 = F.relu(self.l5(q2))
           q2 = self.l6(q2)
   
           return q1, q2
   ```
   
3. 计算 TD 目标：
   $$
   \hat{y}_j=r_j+\gamma\cdot\min\{\hat{q}_{1,j+1}^-,\hat{q}_{2,j+1}^-\}
   $$
   ```python
   # 计算 TD 目标。
   target_Q = torch.min(target_Q1, target_Q2)
   target_Q = r + (~dw) * self.gamma * target_Q  # dw: die or win
   ```
   
4. 让两个价值网络做预测：
   $$
   \hat{q}_{1,j}=q(s_j,a_j;w_\text{1,now}) \quad \text{和} \quad
   \hat{q}_{2,j}=q(s_j,a_j;w_\text{2,now})
   $$
   ```python
   # 让两个价值网络做预测
   current_Q1, current_Q2 = self.q_critic(s, a)
   ```
   
5. 计算 TD 误差：
   $$
   \delta_{1,j}=\hat{q}_{1,j+1}^- - \hat{y}_{j} \quad \text{和} \quad
   \delta_{2,j}=\hat{q}_{2,j+1}^- - \hat{y}_{j}
   $$
   ```python
   # 计算 TD 误差
   q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
   ```
   
6. 更新价值网络：
   $$
   w_{\text{1,new}} \leftarrow w_{\text{1,now}}
   \alpha\cdot\delta_{1,j}\cdot\nabla_w q(s_j,a_j;w_\text{1,now}) \quad
   \text{和} \quad w_{\text{2,new}} \leftarrow w_{\text{2,now}}
   \alpha\cdot\delta_{2,j}\cdot\nabla_w q(s_j,a_j;w_\text{2,now})
   $$
   ```python
   self.q_critic_optimizer.zero_grad()
   q_loss.backward()
   self.q_critic_optimizer.step()
   ```
   
7. 每隔 $k$ 轮更新一次策略网络和三个目标网络：
   ```python
   # 如果延迟计数器大于延迟频率，更新演员网络。
   if self.delay_counter > self.delay_freq:
   ```
   1. 让策略网络做预测：$a_{j} = \mu(s_{};\theta_{\text{now}}) + \epsilon$。然后更新策略网络：
      $$
      \theta_\text{new}\leftarrow\theta_\text{now}+\beta\cdot\nabla_{\theta}\mu(s_j;\theta_\text{now})\cdot\nabla_{a}q(s_j,\hat{a}_j;w_{1,now})
      $$
      ```python
      a_loss = -self.q_critic.Q1(s, self.actor(s)).mean()
      self.actor_optimizer.zero_grad()
      a_loss.backward()
      self.actor_optimizer.step()
      ```
   
   2. 更新目标网络的参数：
      $$
      \begin{align}
      \theta_{\text{new}}^{-} &\leftarrow \tau\cdot \theta_{\text{new}} + (1 - \tau)\cdot \theta_{\text{now}}^{-}
      \\
      w_\text{1,new}^{-} &\leftarrow \tau\cdot w_\text{1,new}+(1-\tau)\cdot w_\text{1,now}^{-}
      \\
      w_\text{2,new}^{-} &\leftarrow \tau\cdot w_\text{2,new}+(1-\tau)\cdot w_\text{2,now}^{-}
      \end{align}
      $$
      ```python
      with torch.no_grad():
      	for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
      		target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
      
      	for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
              target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
      ```
<div align=center>
<img width="1200" src="./png/TD3.svg"/>
</div>
<div align=center>TD3训练流程</div>
