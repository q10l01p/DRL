# PPO
PPO与TRPO的目标相同：在使用**当前数据**的情况下，我们如何能够在策略上迈出尽可能大的改进步骤，但不会走得太远以至于意外地导致性能崩溃？

TRPO尝试通过复杂的**二阶方法**解决这个问题，而PPO是**一阶方法**，使用一些其他技巧来确保新策略接近旧策略。

PPO方法在实施上显著简单，并且与TRPO一样表现良好。

有两个主要的PPO变体：**PPO-Penalty**和**PPO-Clip**。

- PPO 是一种**同策略**算法
- PPO 可用于离散或连续空间的环境
- Spinning Up实现的PPO支持使用MPI进行并行化。

## PPO-Penalty
**PPO-Penalty**大致上解决了像TRPO一样的KL受限更新，但是在目标函数中对KL散度进行惩罚，而不是将其作为硬约束，并且在训练过程中自动调整惩罚系数，以便适当地进行缩放。

## PPO-Clip
**PPO-Clip**在目标函数中没有KL散度项，也没有任何约束。相反，它依赖于目标函数中的专门**剪切**，以消除新策略远离旧策略的动机。

PPO-Clip通过以下方式更新策略：

$$\begin{align}\theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{\mathbb{E}}\left[ L(s,a,\theta_k, \theta)\right]\tag{1}\end{align}$$

通常采用**多步（通常是小批量）SGD**来最大化这个目标。

$L$由以下公式给出：

$$\begin{align}J_{\text{PPO}2}^{\theta^\kappa}(\theta)\approx\sum_{(s_{t},a_{t})} \operatorname*{min}\left(\frac{p_{\theta}\left(a_{t}|s_{t}\right)}{p_{\theta^{k}}\left(a_{t}|s_{t}\right)}A^{\theta^{k}}\left(s_{t},a_{t}\right),\right.\left.\mathrm{clip}\left(\frac{p_{\theta}\left(a_{t}|s_{t}\right)}{p_{\theta^{k}}\left(a_{t}|s_{t}\right)},1-\varepsilon,1+\varepsilon\right)A^{\theta^{k}}\left(s_{t},a_{t}\right)\right)\tag{2}\end{align}$$

其中 

- 操作符（operator）min 是在第一项与第二项里面选择比较小的项。

- 第二项前面有一个裁剪（clip）函数，裁剪函数是指，在括号里面有3项，如果第一项小于第二项，那就输出 $1 - \epsilon$；第一项如果大于第三项，那就输出 $1 + \epsilon$。 

- $\epsilon$ 是一个（较小的）超参数，大致表示**新策略允许离旧策略有多远**，可以设置成 $0.1$或 $0.2$ 。

假设设置$\epsilon = 0.2$，我们可得 

$$\operatorname{clip}\left(\frac{p_\theta\left(a_t|s_t\right)}{p_{\theta^k}\left(a_t|s_t\right)},0.8,1.2\right)$$

如果 $\frac{p_\theta(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}$ 算出来小于 0.8, 那就输出 0.8; 如果算出来大于 1.2, 那就输出1.2。

$$\operatorname{clip}\left(\frac{p_\theta\left(a_t|s_t\right)}{p_{\theta^k}\left(a_t|s_t\right)},1-\varepsilon,1+\varepsilon\right)$$

下图的横轴代表$\frac{p_\theta(a_t|s_t)}{p_{\theta k}(a_t|s_t)}$,纵轴代表裁剪函数的输出。

- 如果$ \frac{p_\theta(a_t|s_t)}{p_{ak}(a_t|s_t)} $大于$1+\varepsilon$输出就是 $1+\varepsilon$;
- 如果小于$1-\varepsilon$，输出就是$1-\varepsilon$；
- 如果介于$1+\varepsilon \sim 1- \varepsilon$,输出等于输入。

<div align=center>
<img width="550" src="https://datawhalechina.github.io/easy-rl/img/ch5/5.2.png "/>
</div>
<div align=center>裁剪函数</div>

如图a 所示，$\frac{p_\theta(a_t|s_t)}{p_{\phi^k}(a_t|s_t)}$是绿色的线； $\operatorname{clip}\left(\frac{p_\theta(a_t/s_t)}{p_{\phi^k}(a_t|s_t)},1-\varepsilon,1+\varepsilon\right)$ 是蓝色的线； 在绿色的线与蓝色的线中间，我们要取一个最小的结果。

假设前面乘上的项$A$ 大于 0, 取最小的结果，就是红色的这条线。

如图b 所示，如果 $A$ 小于 0, 取最小结果的以后，就得到红色的这条线。

<div align=center>
<img width="550" src="https://datawhalechina.github.io/easy-rl/img/ch5/5.3.png "/>
</div>
<div align=center>A对裁剪函数输出的影响</div>

虽然式2看起来有点儿复杂,但实现起来是比较简单的,因为式2想要做的就是希望$p_{\theta(}(a_{t}|s_{t})$与$p_{\varphi(}(a_{t}|s_{t})$比较接近,也就是做示范的模型与实际上学习的模型在优化以后不要差距太大。

怎么让它做到不要差距太大呢？

- 如果$A>0$,也就是某一个状态-动作对是好的，我们希望增大这个状态-动作对的概率。也就是，我们想让$p_\theta(a_t|s_t)$ 越大越好，但它与$p_{\theta^k}(a_t|s_t)$ 的比值不可以超过 $1+\varepsilon$。如果超过 $1+\varepsilon$ ,就没有好处了。红色的线就是目标函数，我们希望目标函数值越大越好，我们希望$p_\theta(a_t|s_t)$ 越大越好。但是$\frac{p_0(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}$只要大过$1+\varepsilon$,就没有好处了。所以在训练的时候，当$p_{\theta^k}(a_t|s_t)$被训练到$\frac{p_\theta(\alpha_t)(s_t)}{p_{t^k}(\alpha_t)}>1+\varepsilon$ 时，它就会停止。假设 $p_\theta(a_t|s_t)$ 比 $p_{\theta^k}(a_t|s_t)$ 还要小，并且这个优势是正的。因为这个动作是好的，我们希望这个动作被采取的概率越大越好，希望 $p_\theta(a_t|s_t)$ 越大越好。所以假设 $p_\theta(a_t|s_t)$ 还比$p_{\theta^k}(a_t|s_t)$ 小，那就尽量把它变大，但只要大到 $1+\varepsilon$ 就好。
- 如果$A<0$,也就是某一个状态-动作对是不好的，那么我们希望把$p_\theta(\alpha_t|s_t)$ 减小。如果$p_\theta(\alpha_t|s_t)$ 比$p_{\theta^k}(a_t|s_t)$还大，那我们就尽量把它减小，减到$\frac{p_\theta(a_t|s_t)}{p_{\theta^k}(a_t|s_t)}$是$1-\varepsilon$ 的时候停止，此时不用再减得更小。

这样的好处就是,我们不会让$p_\theta(a_t|s_t)$与$p_{\theta^k}(a_t|s_t)$差距太大。

式2一个相当复杂的表达式，一眼看上去很难理解它在做什么，或者它如何帮助保持新策略接近旧策略。

事实上，这个目标函数有一个[相当简化的版本](https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view "请查看有关PPO-Clip目标简化形式推导的说明。")，稍微容易理解一些（也是我们在代码中实现的版本）：

$$L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), \;\; g(\epsilon, A^{\pi_{\theta_k}}(s,a)) \right),$$

其中

$$g(\epsilon, A) = \left\{ \begin{array}{ll} (1 + \epsilon) A & A \geq 0 \\ (1 - \epsilon) A & A < 0. \end{array} \right.$$

到目前为止，我们看到剪切作为正则化器，通过消除使策略发生显著变化的动机，而超参数 $\epsilon$ 对应于新策略可以离旧策略多远而仍然对目标有利。

>尽管这种剪切方法在很大程度上确保了合理的策略更新，但仍然有可能得到一个与旧策略相距太远的新策略。不同的PPO实现使用了许多技巧来防止这种情况发生。在我们这里的实现中，我们采用了一种特别简单的方法：**提前停止**。**如果新策略与旧策略之间的平均KL散度超过了阈值，我们就停止梯度更新步骤**。

# 伪代码——PPO-Clip
输入：初始策略参数 $\theta_0$，初始值函数参数 $\phi_{0}$

1. for $k=0,1,2,...$ do
    1. 运行策略 $\pi_k=\pi(\theta_k)$ 在环境中，收集轨迹集合 $D_k=\{\tau_i\}$
    2. 计算回报至终点 $\hat{R}_t.$
    3. 基于当前值函数 $V_\mathrm{\phi k}$，计算优势估计 $\hat{A}_t$（使用任何优势估计方法）。
    4. 通过最大化 PPO-Clip 目标更新策略：
       $$\theta_{k+1}=\arg\max_{\theta}\frac{1}{|\mathcal{D}_{k}|T}\sum_{\tau\in\mathcal{D}_{k}}\sum_{t=0}^{T}\min\left(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{k}}(a_{t}|s_{t})}\hat{A}^{\pi_{\theta}}(s_{t},a_{t}),\:g(\epsilon,\hat{A}^{\pi_{\theta}}_{k}(s_{t},a_{t}))\right),$$
        通常使用 Adam 随机梯度上升法。

    6. 通过均方误差回归拟合值函数：
       $$\phi_{k+1}=\arg\min_{\phi}\frac{1}{|\mathcal{D}_{k}|T}\sum_{\tau\in\mathcal{D}_{k}}\sum_{t=0}^{T}\left(V_{\phi}(s_{t})-\hat{R}_{t}\right)^{2},$$
       通常使用某种梯度下降算法。

2. end for

