# 策略梯度(Policy Gradient)

## 大坑

学到这里，我们先回顾一下，我们是怎样从0到DQN的。

一开始，我们对这个世界一无所知。我们发现我们每走一步，都有一个reward，我们希望我们能够获得更多的reward，所以我们在每个状态或者动作都做上记录，希望统计一下，从这个state出发，我能获得多少奖励。这样我们就可以知道我们应该走那条路。

于是，我们从动态规划到蒙特卡罗，到TD到Qleaning再到DQN，一路为计算Q值和V值绞尽脑汁。但大家有没有发现，我们可能走上一个固定的思维，就是我们的学习，一定要算Q值和V值，往死里算。但算Q值和V值并不是我们最终目的呀，我们要找一个策略，能获得最多的奖励。我们可以抛弃掉Q值和V值么？

答案是，可以，策略梯度(Policy Gradient)算法就是这样以一个算法。

## Policy Gradient

- DQN是一个TD+神经网络的算法
- PG是一个蒙地卡罗+神经网络的算法

我们用 $\pi$ 表示策略，也就是动作的分布。那么我们期望有这么一个magic函数，当我输入state的时候，他能输出 $\pi$，告诉智能体这个状态，应该如何应对。

- 如果智能体的动作是对的，那么就让这个动作获得更多被选择的几率;
- 如果这个动作是错的，那么这个动作被选择的几率将会减少。

怎么衡量对和错呢？PG的想法非常简单粗暴：**蒙地卡罗的G值**！

### 蒙地卡罗

我们从某个state出发，然后一直走，直到最终状态。然后我们从最终状态原路返回，对每个状态评估G值。

所以G值能够表示在策略 $\pi$ 下，智能体选择的这条路径的好坏。

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-f021b528bdd81b4cbfb711c2b71ea244_720w.jpg "/>
</div>
<div align=center></div>

[如何用蒙地卡罗方法（Monte-Carlo）估算V值](./Monte-Carlo.md)

## 直观感受PG算法

我们先用数字，直观感受一下PG算法。

从某个state出发，可以采取三个动作。

假设当前智能体对这一无所知，那么，可能采取平均策略 0 = [33%,33%,33%]。

智能体出发，选择动作A，到达最终状态后开始回溯，计算得到 G = 1。

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-b0252a3e09676a946e38ff3c8c3a4431_720w.jpg "/>
</div>
<div align=center></div>

我们可以更新策略，因为该路径选择了A而产生的，并获得G = 1；因此我们要更新策略：让A的概率提升，相对地，BC的概率就会降低。 计算得新策略为： 1 = [50%,25%,25%]

虽然B概率比较低，但仍然有可能被选中。第二轮刚好选中B。

智能体选择了B，到达最终状态后回溯，计算得到 G = -1。

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-5a6cf01c21f4b27c2c8a3828a4b45bf4_720w.jpg "/>
</div>
<div align=center></div>

所以我们对B动作的评价比较低，并且希望以后会少点选择B，因此我们要降低B选择的概率，而相对地，AC的选择将会提高。

计算得新策略为： 2 = [55%,15%,30%]

最后随机到C，回溯计算后，计算得G = 5。

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-b097923d75bb7559bb7d2efd14c16210_720w.webp "/>
</div>
<div align=center></div>

C比A还要多得多。因此这一次更新，C的概率需要大幅提升，相对地，AB概率降低。 3 = [20%,5%,75%]

## 示例代码分析

### 更新框架

```python
for i_episode in range(int(num_episodes)):
    episode_return = 0  # 初始化回合累计奖励
    transition_dict = {  # 用于存储当前回合的经验信息的字典
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    
    state = env.reset()  # 重置环境并获取初始状态
    done = False  # 重置回合结束标志
    
    # 在当前回合循环执行每个时间步
    while not done:  # 时间步循环开始
        action = agent.take_action(state)  # 代理选择动作
        next_state, reward, done, _ = env.step(action)  # 执行动作并获取下一个状态、奖励和回合结束信息
        
        # 将经验信息添加到transition_dict中
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(reward)
        
        state = next_state  # 更新当前状态
        episode_return += reward  # 累计回合奖励

    return_list.append(episode_return)  # 记录当前回合的累计奖励
```

1. 开始一场游戏，并重制state
2. 根据state，选择action
3. 把action代入环境，获得observation_, reward, done, info
4. 记录数据
5. 计算G值，并开始学习策略。

注意:

1. 记录数据 在**DQN**，我们记录data = (s,a,r,s_,d) 五要素。并且我们记录在一个队列，需要用到的时候，从队列中**随机抽取**。 但**PG**中，我们记录的data = (s,a,r) 三要素就可以了，我们的记录是**不能打乱**的。因为当我们要计算G值的时候，我们需要从后往前回溯计算。
2. 清除数据 这些记录是**用完即弃**，可以看到，在learn函数最后，智能体学习完数据后，就会清空列表。

### 计算G值

经过一次游戏，到结束状态，我们计算所有经过的state的G值。

示例代码中，函数`_discount_and_norm_rewards()`就是计算G值。

函数分为两部分，一部分计算G值，一部分把G值进行归一化处理。

```PYTHON
def _discount_and_norm_rewards(self, transition_dict):
    """
    计算折扣回报并对其进行归一化处理。

    返回:
        np.ndarray: 归一化后的折扣回报数组
    """
    # 初始化折扣回报数组
    G_list = np.zeros_like(reward_list)
    reward_list = transition_dict['rewards']
    
    # 计算折扣回报（G值）
    G = 0
    for i in reversed(range(len(reward_list))):  # 从最后一步算起
        G = G * self.gamma + reward_list[i]
         G_list[i] = G

    # 归一化处理（均值为0，标准差为1）
    G_list -= np.mean(G_list)
    G_list /= np.std(G_list)

    # 返回归一化后的折扣回报数组
    return G_list
```

如图，假设我们经过6个state，到达最终状态，获得收获如下。

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-9f7bbfc8d7cf4e7424f9dcd9d5ae4be1_720w.webp "/>
</div>
<div align=center></div>

1. 首先我们创建一个全零向量，其大小和储存reward的列表相同。
2. 我们从后往前计算，在这个例子中，就是从state6开始。所以我们用了一个反向循环的方式来计算。
3. 每次循环，把上一个G值乘以折扣(gamma) ，然后加上这个state获得的reward即可。我们把这个值记录在`G_list`

```PYTHON
    # 初始化折扣回报数组
    G_list = np.zeros_like(reward_list)
    
    # 计算折扣回报（G值）
    G = 0
    for i in reversed(range(len(reward_list))):  # 从最后一步算起
        G = G * self.gamma + reward_list[i]
         G_list[i] = G
```

我们可以用G值直接进行学习，但一般来说，对数据进行**归一化**处理后，训练效果会更好。我们只需要简单减去平均数，除以方差即可。

```PYTHON
    # 归一化处理（均值为0，标准差为1）
    G_list -= np.mean(G_list)
    G_list /= np.std(G_list)
```

### 带权重的梯度下降

```PYTHON
def update(self, transition_dict, G_list):
    """
    使用REINFORCE算法更新策略网络的参数。

    参数:
        transition_dict (dict): 包含经验信息的字典，包括rewards、states和actions

    返回:
        无返回值
    """
    # 从transition_dict中获取经验信息
    state_list = transition_dict['states']
    action_list = transition_dict['actions']

    self.optimizer.zero_grad()  # 梯度清零

    # 从最后一步开始逆序遍历经验
    for i in reversed(range(len(reward_list))):
        state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
        action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
        log_prob = torch.log(self.policy_net(state).gather(1, action))
        
        # 计算每一步的损失函数
        loss = -log_prob * G_list[i]
        
        # 反向传播计算梯度
        loss.backward()

    # 使用梯度下降进行参数更新
    self.optimizer.step()
```

我们某一个状态为例，在某个状态下，通过网络预测（_logits），真实值（ep_as），G值（discounted_ep_rs_norm）如下图

<div align=center>
<img width="800" src=" https://pic3.zhimg.com/80/v2-5bd5d1c618fe38234aaf1155b58876b6_720w.webp "/>
</div>
<div align=center></div>

我们可以把这个过程想象成一个分类任务。在训练的时候，只有真实值为1，其他为0。所以动作1,3,4的概率将会向0靠，也就是减少。而动作2的概率将会向1靠，也就是说会有所提升。

那么G值的意义是什么呢？

loss是根据G值进行调整的。当G值调整大小的时候，相当于每次训练幅度进行调整。例如G值为2，那么调整的幅度将会是1的两倍。

<div align=center>
<img width="800" src=" https://pic4.zhimg.com/80/v2-f1e5294baca85eb8862e6336f5cb0f8f_720w.webp "/>
</div>
<div align=center></div>

如果G值是一个负数呢，那么相当于我们进行反向的调整。如下图，如果G值为-1，那么说明选择动作2并不是一个“明智”的动作。于是我们让这个动作2的预测值降低，相当于“远离”真实值1。而其他动作的概率有所提升，相当于“远离”真实值0。

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-d3039ebb9f7111fd2fcb7c619fe42e6c_720w.webp "/>
</div>
<div align=center></div>

## 结论

PG用一个全新的思路解决了问题。

但实际效果显得不太稳定，在某些环境下学习较为困难。

另外由于采用了MC的方式，需要走到最终状态才能进行更新，而且只能进行一次更新，这也是GP算法的效率不高的原因。
