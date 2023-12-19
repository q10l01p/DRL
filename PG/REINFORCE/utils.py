import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import os, random

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

# 定义一个神经网络模型，继承自torch.nn.Module
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 调用父类构造函数
        super(PolicyNet, self).__init__()
        
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.dropout = torch.nn.Dropout(p=0.0)

        self.apply(init_weight)

    # 定义前向传播方法
    def forward(self, x):
        # 使用ReLU激活函数对第一个全连接层的输出进行非线性变换
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 使用ReLU激活函数对第二个全连接层的输出进行非线性变换
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 使用softmax激活函数对第二个全连接层的输出进行非线性变换，dim=1表示对每一行进行softmax操作
        return F.softmax(self.fc3(x), dim=1)

def evaluate_policy(env, agent, steps, turns):
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
        while not done:
            frames.append(env.render())
            # 在测试时采取确定性的动作
            a = agent.choose_action(s)
            s_next, r, dw, tr, _ = env.step(a)  # 执行动作，获取下一个状态、奖励和其他信息
            done = (dw or tr)  # 判断是否完成（done为True）

            total_scores += r  # 累加每步的奖励到总分
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
            a = agent.choose_action(s)
            s_next, r, dw, tr, info = env.step(a)  # 执行动作，获取下一个状态、奖励和其他信息
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


def all_seed(env, seed):
    """
    设置随机种子以确保实验的可重复性

    参数:
    - env: Gym 环境，用于训练模型
    - seed: 随机种子值

    说明:
    1. 使用给定的随机种子设置 NumPy、Python、PyTorch 和 CUDA 的随机生成器。
    2. 禁用 CUDA 的非确定性操作以确保实验结果的一致性。
    """

    np.random.seed(seed)  # 设置 NumPy 随机种子
    random.seed(seed)  # 设置 Python 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch CUDA 随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python Hash 随机种子
    torch.backends.cudnn.deterministic = True  # 禁用 CUDA 非确定性操作以确保实验结果的一致性
    torch.backends.cudnn.benchmark = False  # 禁用 CUDA 非确定性操作以确保实验结果的一致性
    torch.backends.cudnn.enabled = False  # 禁用 CUDA 非确定性操作以确保实验结果的一致性