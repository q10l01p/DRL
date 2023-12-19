import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from torch.distributions import Beta,Normal


class BetaActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        """
        Beta分布策略网络，输出alpha和beta参数

        参数:
        - state_dim: 状态空间维度
        - action_dim: 动作空间维度
        - net_width: 神经网络宽度

        说明:
        1. 初始化网络结构，包括三个全连接层。
        2. alpha_head 用于输出alpha参数。
        3. beta_head 用于输出beta参数。
        """
        super(BetaActor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)  # 1. 第一个全连接层
        self.l2 = nn.Linear(net_width, net_width)   # 1. 第二个全连接层
        self.alpha_head = nn.Linear(net_width, action_dim)  # 2. 输出alpha参数的全连接层
        self.beta_head = nn.Linear(net_width, action_dim)  # 3. 输出beta参数的全连接层

    def forward(self, state):
        """
        正向传播

        参数:
        - state: 输入的状态

        返回:
        - alpha: Beta分布的alpha参数
        - beta: Beta分布的beta参数

        说明:
        1. 通过两个双曲正切激活函数传播信号。
        2. 使用软正切激活函数计算并输出alpha和beta参数，确保参数为正数。
        """
        a = torch.tanh(self.l1(state))  # 1. 第一个双曲正切激活函数
        a = torch.tanh(self.l2(a))  # 1. 第二个双曲正切激活函数

        # 使用软正切激活函数得到 alpha 和 beta 参数，确保为正数
        alpha = F.softplus(self.alpha_head(a)) + 1.0  # 2. 计算alpha参数
        beta = F.softplus(self.beta_head(a)) + 1.0  # 2. 计算beta参数

        return alpha, beta

    def get_dist(self, state):
        """
        获取动作分布

        参数:
        - state: 输入的状态

        返回:
        - dist: Beta分布

        说明:
        获取Beta分布，由alpha和beta参数构建。
        """
        alpha, beta = self.forward(state)  # 获取alpha和beta参数
        dist = Beta(alpha, beta)  # 构建Beta分布
        return dist

    def deterministic_act(self, state):
        """
        根据状态进行确定性动作选择

        参数:
        - state: 输入的状态

        返回:
        - mode: Beta分布的众数

        说明:
        直接返回Beta分布的众数，用于确定性动作选择。
        """
        alpha, beta = self.forward(state)  # 获取alpha和beta参数
        mode = (alpha) / (alpha + beta)  # 计算众数
        return mode


class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        """
        高斯策略网络，输出动作均值和标准差

        参数:
        - state_dim: 状态空间维度
        - action_dim: 动作空间维度
        - net_width: 神经网络宽度

        说明:
        1. 初始化网络结构，包括三个全连接层。
        2. mu_head 用于输出动作均值。
        3. sigma_head 用于输出动作标准差。
        """
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)  # 1. 第一个全连接层
        self.l2 = nn.Linear(net_width, net_width)   # 1. 第二个全连接层
        self.mu_head = nn.Linear(net_width, action_dim)  # 2. 输出动作均值的全连接层
        self.sigma_head = nn.Linear(net_width, action_dim)  # 3. 输出动作标准差的全连接层

    def forward(self, state):
        """
        正向传播

        参数:
        - state: 输入的状态

        返回:
        - mu: 动作均值
        - sigma: 动作标准差

        说明:
        1. 通过两个双曲正切激活函数传播信号。
        2. 通过 sigmoid 函数计算并输出动作均值 mu。
        3. 通过 softplus 函数计算并输出动作标准差 sigma。
        """
        a = torch.tanh(self.l1(state))  # 1. 第一个双曲正切激活函数
        a = torch.tanh(self.l2(a))  # 1. 第二个双曲正切激活函数

        # 获取均值（mu）和标准差（sigma）
        mu = torch.sigmoid(self.mu_head(a))  # 2. 计算动作均值
        sigma = F.softplus(self.sigma_head(a))  # 3. 计算动作标准差

        return mu, sigma

    def get_dist(self, state):
        """
        获取动作分布

        参数:
        - state: 输入的状态

        返回:
        - dist: 动作分布

        说明:
        获取正态分布，由动作均值和标准差构建。
        """
        mu, sigma = self.forward(state)  # 获取均值和标准差
        dist = Normal(mu, sigma)  # 构建动作分布
        return dist

    def deterministic_act(self, state):
        """
        根据状态进行确定性动作选择

        参数:
        - state: 输入的状态

        返回:
        - 动作均值

        说明:
        直接返回动作均值，用于确定性动作选择。
        """
        mu, _ = self.forward(state)  # 获取动作均值
        return mu


class GaussianActor_mu(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        """
        高斯策略网络，输出动作均值

        参数:
        - state_dim: 状态空间维度
        - action_dim: 动作空间维度
        - net_width: 神经网络宽度
        - log_std: 初始动作标准差的对数值，默认为0

        说明:
        1. 初始化网络结构，包括三个全连接层。
        2. mu_head 用于输出动作均值。
        3. 初始设置 mu_head 的权重和偏置。
        4. action_log_std 为动作标准差的对数值，通过学习得到。
        """
        super(GaussianActor_mu, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)  # 1. 第一个全连接层
        self.l2 = nn.Linear(net_width, net_width)   # 1. 第二个全连接层
        self.mu_head = nn.Linear(net_width, action_dim)  # 2. 输出动作均值的全连接层
        self.mu_head.weight.data.mul_(0.1)  # 3. 初始化权重
        self.mu_head.bias.data.mul_(0.0)  # 3. 初始化偏置

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)  # 4. 动作标准差的对数值

    def forward(self, state):
        """
        正向传播

        参数:
        - state: 输入的状态

        返回:
        - mu: 动作均值

        说明:
        1. 通过两个 ReLU 激活函数传播信号。
        2. 通过 sigmoid 函数计算并输出动作均值 mu。
        """
        a = torch.relu(self.l1(state))  # 1. 第一个激活函数
        a = torch.relu(self.l2(a))  # 1. 第二个激活函数
        mu = torch.sigmoid(self.mu_head(a))  # 2. 计算动作均值
        return mu

    def get_dist(self, state):
        """
        获取动作分布

        参数:
        - state: 输入的状态

        返回:
        - dist: 动作分布

        说明:
        1. 获取动作均值 mu。
        2. 根据学习得到的动作标准差，构建动作分布。
        """
        mu = self.forward(state)  # 1. 获取动作均值
        action_log_std = self.action_log_std.expand_as(mu)  # 扩展动作标准差维度
        action_std = torch.exp(action_log_std)  # 计算动作标准差
        dist = Normal(mu, action_std)  # 构建动作分布
        return dist

    def deterministic_act(self, state):
        """
        根据状态进行确定性动作选择

        参数:
        - state: 输入的状态

        返回:
        - 动作均值

        说明:
        直接返回动作均值，用于确定性动作选择。
        """
        return self.forward(state)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        """
        初始化 Actor 模型

        参数:
        - state_dim: 输入状态的维度
        - action_dim: 输出动作的维度
        - net_width: 神经网络中间层的宽度

        说明:
        1. 创建 Actor 模型，用于确定在给定状态下采取的动作的概率分布。
        2. 包含三个全连接层，分别是输入层（state_dim -> net_width）、中间层（net_width -> net_width）和输出层（net_width -> action_dim）。
        """
        super(Actor, self).__init__()

        # 定义神经网络的三个全连接层
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        """
        定义前向传播过程

        参数:
        - state: 输入状态

        返回:
        - n: 经过前向传播后的输出

        说明:
        1. 使用 tanh 激活函数的前向传播。
        """
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim=0):
        """
        定义策略函数，输出动作的概率分布

        参数:
        - state: 输入状态
        - softmax_dim: 对哪个维度进行 softmax 操作，默认为第 0 维

        返回:
        - prob: 经过 softmax 处理后的动作概率分布

        说明:
        1. 利用前向传播的结果计算输出动作的概率分布。
        2. 使用 softmax 函数对输出进行处理，得到归一化的概率分布。
        """
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob


class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        """
        初始化 Critic 模型

        参数:
        - state_dim: 输入状态的维度
        - net_width: 神经网络中间层的宽度

        说明:
        1. 创建 Critic 模型，用于估算给定状态的值函数（Q值或者状态值）。
        2. 包含三个全连接层，分别是输入层（state_dim -> net_width）、中间层（net_width -> net_width）和输出层（net_width -> 1）。
        """
        super(Critic, self).__init__()

        # 定义神经网络的三个全连接层
        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        """
        定义前向传播过程

        参数:
        - state: 输入状态

        返回:
        - v: 经过前向传播后的值函数估计值

        说明:
        1. 使用 ReLU 激活函数的前向传播。
        2. 输出为值函数的估计值，表示给定状态的Q值或者状态值。
        """
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


def evaluate_policy(env, agent, turns, cfg):
    """
    评估策略的函数

    参数:
    - env: 环境对象，代表了模拟的环境
    - agent: 代理对象，执行动作和选择动作的智能体
    - turns: 循环轮数，默认为3

    返回:
    返回通过评估得到的总分的平均值（整数）

    说明:
    1. 通过在模拟环境中执行智能体的策略，计算多次评估的平均分数。
    2. 在测试阶段，采取确定性的动作。
    3. total_scores 记录每轮评估的总分。
    4. 返回平均分数（整数）。
    """
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()  # 重置环境，获取初始状态和信息
        done = False
        frames = []
        i = 0
        while not done:
            frames.append(env.render())
            # 在测试时采取确定性的动作
            a, logprob_a = agent.select_action(s, deterministic=True)
            act = action_adapter(a, cfg['max_action'])
            s_next, r, dw, tr, info = env.step(act)  # 执行动作，获取下一个状态、奖励和其他信息
            done = (dw or tr)  # 判断是否完成（done为True）

            total_scores += r  # 累加每步的奖励到总分
            i += 1
            s = s_next  # 更新状态为下一个状态

    return int(total_scores / turns)  # 返回总分的平均值（取整）


def test_policy(env, agent, steps, turns, path, cfg):
    """
    评估策略的函数

    参数:
    - env: 环境对象，代表了模拟的环境
    - agent: 代理对象，执行动作和选择动作的智能体
    - turns: 循环轮数，默认为3

    返回:
    返回通过评估得到的总分的平均值（整数）

    说明:
    1. 通过在模拟环境中执行智能体的策略，计算多次评估的平均分数。
    2. 在测试阶段，采取确定性的动作。
    3. total_scores 记录每轮评估的总分。
    4. 返回平均分数（整数）。
    """
    total_scores = 0
    for j in range(turns):
        frames = []
        s, info = env.reset()  # 重置环境，获取初始状态和信息
        done = False
        while not done:
            frames.append(env.render())
            # 在测试时采取确定性的动作
            a, logprob_a = agent.select_action(s, deterministic=True)
            act = action_adapter(a, cfg['max_action'])
            s_next, r, dw, tr, info = env.step(act)  # 执行动作，获取下一个状态、奖励和其他信息
            done = (dw or tr)  # 判断是否完成（done为True）

            total_scores += r  # 累加每步的奖励到总分
            s = s_next  # 更新状态为下一个状态

        display_frame_as_gif(frames, steps, total_scores, path, cfg)


def str2bool(v):
    """
    将字符串转换为布尔值，用于 argparse

    参数:
    - v: 输入字符串

    返回:
    布尔值，表示字符串的真假

    说明:
    1. 用于将字符串转换为布尔值，主要用于解析命令行参数。
    2. 如果输入是布尔类型，则直接返回。
    3. 如果输入是可识别的字符串表示（如 'yes', 'True'），返回对应的布尔值。
    4. 如果输入是不可识别的字符串表示，打印错误信息并引发异常。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise


def display_frame_as_gif(frames, total_steps, rewards, path, cfg):
    env_id = cfg['env_name']
    algo_name = cfg['algo_name']
    video_path = f"videos/{path}"
    plt.figure(figsize=(frames[0].shape[1]/72, frames[0].shape[0]/72), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        plt.title(f"{env_id}_{algo_name}_{rewards}_{i}")
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    try:
        os.makedirs(video_path)
    except FileExistsError:
        pass

    print("保存视频")
    anim.save(f'{video_path}/{total_steps}_{rewards}.gif', writer='pillow')


def action_adapter(a, max_action):
    '''
    将动作从[0, 1]适配到[-max_action, max_action]

    参数:
    - a: 输入的动作值，范围在[0, 1]
    - max_action: 允许的最大动作值

    返回:
    - 适配后的动作值，范围在[-max_action, max_action]

    说明:
    将输入的动作从[0, 1]的范围适配到[-max_action, max_action]的范围。
    '''
    return 2 * (a - 0.5) * max_action


def Reward_adapter(r, env_name):
    """
    奖励适配器函数，根据环境索引对奖励进行适配

    参数:
    - r: 原始奖励值
    - EnvIndex: 环境索引

    返回:
    - 适配后的奖励值

    说明:
    - 对于 BipedalWalker 环境（EnvIndex 为 0 或 1）：如果奖励小于等于 -100，则将奖励设为 -1。
    - 对于 Pendulum-v0 环境（EnvIndex 为 3）：将奖励映射到区间 [-1, 1]，通过(r + 8) / 8 实现。
    """

    # 对于 BipedalWalker
    if env_name == "Pendulum-v1":
        r = (r + 8) / 8  # 将奖励映射到区间 [-1, 1]
    elif env_name == "BipedalWalker-v3":
        if r <= -100:
            r = -1
    elif env_name == "LunarLanderContinuous-v2":
        r = r
    elif env_name == "MountainCarContinuous-v0":
        r = r * 100
    elif env_name == "HumanoidStandup-v4":
        r = r / 100
    else:
        r = r
    return r