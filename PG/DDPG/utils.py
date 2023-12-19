import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from torch.distributions import Beta,Normal


def init_weight(m):
    """
    初始化神经网络模型权重的函数

    参数:
    - m: 神经网络模型的模块

    说明:
    1. 判断模块是否为线性层（nn.Linear）。
    2. 如果是线性层，使用 Xavier 正态分布初始化权重，偏置初始化为零。
    """
    if isinstance(m, nn.Linear):  # 判断模块是否为线性层
        nn.init.xavier_normal_(m.weight)  # 使用 Xavier 正态分布初始化权重
        nn.init.constant_(m.bias, 0.0)  # 偏置初始化为零


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
            a = agent.select_action(s, deterministic=True)
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
            a = agent.select_action(s, deterministic=True)
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
    