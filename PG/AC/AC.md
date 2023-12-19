# Actor-Critic

PG利用带权重的梯度下降方法更新策略，而获得权重的方法是蒙地卡罗计算G值。

蒙地卡罗需要完成整个游戏过程，直到最终状态，才能通过回溯计算G值。这使得PG方法的效率被限制。

那我们可不可以更快呢？相信大家已经想到了，那就是改为TD。

但改为TD还有一个问题需要解决，就是：在PG，我们需要计算G值；那么在TD中，我们应该怎样估算每一步的Q值呢？神经网络解决！

Actor-Critic，其实是用了两个网络：

两个网络有一个共同点，输入状态S:

- Actor：输出策略，负责选择动作；
- Critic：负责计算每个动作的分数。

大家可以形象地想象为，Actor是舞台上的舞者，Critic是台下的评委。

Actor在台上跳舞，一开始舞姿并不好看，Critic根据Actor的舞姿打分。Actor通过Critic给出的分数，去学习：如果Critic给的分数高，那么Actor会调整这个动作的输出概率；相反，如果Critic给的分数低，那么就减少这个动作输出的概率。

所以依我的观点来看，AC不是PG+DQN，而应该说**AC是TD版本的PG**。

## TD-error

- DQN预估的是Q值
- AC中的Critic，估算的是V值。

直接用network估算的Q值作为更新值，效果会不太好。

原因我们可以看下图：

<div align=center>
<img width="800" src="https://pic1.zhimg.com/80/v2-9bc653d87818a7919abebf049b15dab0_720w.webp"/>
</div>
<div align=center></div>

假设我们用Critic网络，预估到S状态下三个动作A1，A2，A3的Q值分别为1,2,10。

但在开始的时候，我们采用平均策略，于是随机到A1。于是我们用策略梯度的带权重方法更新策略，这里的权重就是Q值。

于是策略会更倾向于选择A1，意味着更大概率选择A1。结果A1的概率就持续升高...

这就掉进了**正数陷阱**。我们明明希望A3能够获得更多的机会，最后却是A1获得最多的机会。

这是为什么呢？

这是因为Q值用于是一个正数，如果权重是一个正数，那么我们相当于提高对应动作的选择的概率。权重越大，我们调整的幅度将会越大。

其实当我们有足够的迭代次数，这个是不用担心这个问题的。因为总会有机会抽中到权重更大的动作，因为权重比较大，抽中一次就能提高很高的概率。

但在强化学习中，往往没有足够的时间让我们去和环境互动。这就会出现由于运气不好，使得一个**很好**的动作没有被采样到的情况发生。

要解决这个问题，我们可以通过减去一个**baseline**，令到权重有正有负。而通常这个baseline，我们选取的是权重的平均值。减去平均值之后，值就变成有正有负了。

而**Q值的期望(均值)就是V**。

<div align=center>
<img width="800" src="https://pic4.zhimg.com/80/v2-bb84b957eee9fcd821a78e5631d5ac57_720w.webp"/>
</div>
<div align=center></div>

所以我们可以得到更新的权重：Q(s,a)-V(s)

随之而来的问题是，这就需要两个网络来估计Q和V了。但马尔科夫告诉我们，很多时候，V和Q是可以互相换算的。

Q(s,a)用 $\gamma * V(s^{\prime}) + r$ 来代替，于是整理后就可以得到:

$$\text{TD-error:}\quad\gamma * V(s^{\prime}) + r - V(s)$$

这个和之前DQN的更新公式非常像:

- DQN的更新用了Q
- TD-error用的是V

如果Critic是用来预估V值，而不是原来讨论的Q值。那么，这个TD-error是用来更新Critic的loss了！

没错，**Critic的任务就是让TD-error尽量小。然后TD-error给Actor做更新。**

总结一下TD-error的知识：

1. 为了避免正数陷阱，我们希望Actor的更新权重有正有负。因此，我们把Q值减去他们的均值V。有：Q(s,a)-V(s)
2. 为了避免需要预估V值和Q值，我们希望把Q和V统一；由于 $Q(s,a) = \gamma * V(s^{\prime}) + r - V(s)$。所以我们得到TD-error公式： $\text{TD-error}=\gamma * V(s^{\prime}) + r - V(s)$
3. TD-error就是Actor更新策略时候，带权重更新中的权重值；
4. 现在Critic不再需要预估Q，而是预估V。而根据马可洛夫链所学，我们知道TD-error就是Critic网络需要的loss，也就是说，Critic函数需要最小化TD-error。

## 算法

1. 定义两个network：Actor 和 Critic
2. 进行N次更新
    1. 从状态s开始，执行动作a，得到奖励r，进入状态s'
    2. 记录的数据
    3. 把输入到Critic，根据公式： $\text{TD-error}=\gamma * V(s^{\prime}) + r - V(s)$ 求 TD-error，并缩小TD-error
    4. 把输入到Actor，计算策略分布。

<div align=center>
<img width="800" src="https://pic1.zhimg.com/80/v2-06c9787f9cd9a71d92ce0bbeb871af60_720w.webp"/>
</div>
<div align=center></div>

## 代码解释

### 更新流程

- 在PG，智能体需要从头一直跑到尾，直到最终状态才开始进行学习。
- 在AC，智能体采用是每步更新的方式。

注意，我们需要**先更新Critic**，并计算出TD-error。再**用TD-error更新Actor**。

在示例代码中，Actor 和 Critic两个网络是完全分离的。

但在实做得时候，很多时候我们会把Actor和Critic**公用网络前面的一些层**。例如state是一张图片，我们可以先通过几层的CNN进行特征的提取，再分别输出Actor的动作概率分布和Critic的V值。

<div align=center>
<img width="800" src="https://pic4.zhimg.com/80/v2-73057fd1b2741df65e47c24be3cfc197_720w.webp"/>
</div>
<div align=center></div>

### 修改reward

```python
if done:
    r = -20
```

意思是：如果已经到达最终状态，那么奖励直接扣20点。这是为什么呢？

首先我们要明确，这个CartPole游戏最终目的，是希望坚持越久越好。所以大家可以想象这么一个过程：在某个濒死状态s下，选择动作a，进入结束状态s，收获r，在CartPole中，这个reward为 1.0。

但我们并不希望游戏结束，我们希望智能体能在濒死状态下“力挽狂澜”！

于是我们把reward减去20，相当于是对濒死状态下，选择动作a的强烈不认同。通过-20大幅减少动作a出现的概率。

再进一步，reward会向前传播，让智能体濒死状态之前状态时，不选择会进入濒死状态的动作，努力避免进入濒死状态。

所以我们说，reward是一个**主观因素**很强的数值。当环境返回的reward不能满足的我们的时候，我们完全可以进行reward的修改，让智能体更快学习。

### Critic 的学习

```python
td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

self.critic_optimizer.zero_grad()
critic_loss.backward()
self.critic_optimizer.step()
```

重点关注td-error的计算。

### Actor 的学习

Actor 的学习本质上是PG的更新，也就是加权的学习。

```python
td_delta = td_target - self.critic(states)
log_probs = torch.log(self.actor(states).gather(1, actions))
actor_loss = torch.mean(-log_probs * td_delta.detach())

self.actor_optimizer.zero_grad()
actor_loss.backward()
self.actor_optimizer.step()
```

## 总结

要掌握AC算法，关键是理解TD-error的来历。不然Critic要最小化TD-error，Actor要把TD-error带参数更新，听上去就会很懵。

理解TD-error本质是Q(s,a)-V(s)来的，而Q(s,a)转为由V(s')+r的形式表示，整个思路就会非常清晰。
