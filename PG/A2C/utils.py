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

        return F.softmax(self.fc3(x), dim=1)
    

class Actor(nn.Module):
    def __init__(self, state_dim, net_width, action_dim):
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
        n = F.relu(self.l1(state))
        n = F.relu(self.l2(n))
        n = F.softmax(self.l3(n), dim=1)
        return n


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
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v

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

def a2c_step(ac_net, optimizer_ac, states, actions, returns, advantages, value_loss_coeff, entropy_coeff):
    """
    更新Actor-Critic网络的一步（A2C算法的单步训练过程）

    参数:
        ac_net (ActorCriticNet): Actor-Critic网络模型
        optimizer_ac (torch.optim.Optimizer): Actor-Critic网络的优化器
        states (torch.Tensor): 输入状态的张量
        actions (torch.Tensor): 执行的动作的张量
        returns (torch.Tensor): 返回值（回报）的张量
        advantages (torch.Tensor): 优势值的张量
        value_loss_coeff (float): 价值损失系数
        entropy_coeff (float): 熵损失系数

    返回:
        dict: 包含actor_critic_loss的字典，用于监控训练过程中的损失
    """
    # 获取动作的对数概率
    log_probs = ac_net.get_log_prob(states, actions)
    
    # 计算策略损失，即对数概率乘以优势值的均值
    policy_loss = -(log_probs * advantages).mean()

    # 获取状态的估值（值函数的输出）
    value = ac_net.get_value(states)
    
    # 计算值函数损失，即均方误差损失
    value_loss = nn.MSELoss()(value, returns)

    # 获取动作的熵（用于增加探索性）
    entropy = ac_net.get_entropy(states)

    # 计算Actor-Critic网络的总损失
    ac_loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy

    # 清零梯度
    optimizer_ac.zero_grad()
    
    # 反向传播计算梯度
    ac_loss.backward()
    
    # 使用梯度裁剪，防止梯度爆炸
    nn.utils.clip_grad_norm_(ac_net.parameters(), 20)
    
    # 使用优化器进行参数更新
    optimizer_ac.step()

    # 返回包含actor_critic_loss的字典，用于监控训练过程中的损失
    return {"actor_critic_loss": ac_loss}

def env_agent_config(cfg, path):
    """
    配置环境和代理

    参数:
    - cfg: 包含配置信息的字典
    - path: 模型保存路径

    返回:
    - env: Gym 环境
    - agent: PPO 代理

    说明:
    1. 创建指定环境并设置渲染模式。
    2. 如果配置中设置了种子，则为环境设置种子。
    3. 获取环境的状态空间维度和动作空间维度。
    4. 更新配置字典中的状态维度和动作维度。
    5. 创建代理。

    注意:
    - 代理的创建依赖于配置信息和模型保存路径。
    """
    # 创建环境
    env = gym.make(cfg['env_name'], render_mode=cfg['render_mode'])
    eval_env = gym.make(cfg['env_name'], render_mode=cfg['render_mode'])

    # 如果配置中设置了种子，则为环境设置种子
    if cfg['seed'] != 0:
        all_seed(env, seed=cfg['seed'])  

    # 获取状态空间维度
    n_states = env.observation_space.shape[0]

    # 获取动作空间维度 
    if type(env.action_space) == Discrete:
        n_actions = env.action_space.n  # 离散动作的数量
    else:
        n_actions = env.action_space.shape[0]  # 连续动作的维度

    # 最大步数
    max_e_steps = env._max_episode_steps 

    # 解除环境限制
    if cfg['unwrap']:  
        env = env.unwrapped 

    if type(env.action_space) == Discrete:
        n_actions = env.action_space.n  # 离散动作的数量
    else:
        n_actions = env.action_space.shape[0]  # 连续动作的维度
        env_continuous = True  # 环境动作是连续的

    # 打印环境信息
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}，最大步数：{max_e_steps}, 是否连续动作：{env_continuous}")

    # 更新n_states和n_actions到cfg参数中
    cfg.update({"state_dim": n_states, "action_dim": n_actions, "max_e_steps": max_e_steps, "env_continuous": env_continuous}) 

    # 创建 PPO 代理
    agent = AC(cfg)
    return env, eval_env, agent