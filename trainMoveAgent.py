# -*- coding: utf-8 -*-
"""
@File    : trainMoveAgent
@Author  : Limit
@Date    : 2025/4/24 16:00
@Description : 使用深度强化学习训练生物体移动
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
import os
import time
import turtle
from collections import deque
import matplotlib.pyplot as plt

from myCreature import *

# 设置随机种子，保证实验可重现
def set_seed(seed):
    """
    设置所有相关随机数生成器的种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 添加状态归一化类
class RunningMeanStd:
    """
    用于状态归一化的运行时均值和标准差计算
    """
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        """更新均值和方差"""
        if np.isnan(x).any() or np.isinf(x).any():
            # 跳过包含NaN或Inf的输入
            return
            
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        """归一化输入"""
        x_safe = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return (x_safe - self.mean) / np.sqrt(self.var + 1e-8)


# 定义Actor网络（策略网络）
class ActorNetwork(nn.Module):
    """
    Actor网络：根据状态输出动作的均值和方差
    用于生成连续动作空间中的动作
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        
        # 构建网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 添加层归一化来增加稳定性
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 使用正确的初始化以提高稳定性
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重以提高稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        前向传播，增加了数值稳定性检查
        """
        # 检查输入是否包含NaN或Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            # 替换NaN和Inf值为0
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        
        # 输出动作分布的均值和标准差
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 0)  # 更严格的标准差限制
        std = torch.exp(log_std)
        
        # 确保输出不含NaN
        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=0.1)
        
        return mean, std
    
    def sample(self, state):
        """
        从策略分布中采样动作
        """
        try:
            mean, std = self.forward(state)
            normal = Normal(mean, std)
            x = normal.rsample()  # 使用重参数化技巧进行采样
            action = torch.tanh(x) * self.max_action  # 使用tanh缩放到[-max_action, max_action]
            
            # 计算log概率，用于PPO算法
            log_prob = normal.log_prob(x)
            # 由于使用tanh转换，需要调整log概率
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            return action, log_prob
        except Exception as e:
            print(f"动作采样错误: {e}")
            # 返回零动作和零对数概率作为后备
            return torch.zeros_like(mean) * self.max_action, torch.zeros((state.shape[0], 1), device=state.device)
    
    def get_action(self, state):
        """
        获取确定性动作，用于评估
        """
        try:
            mean, _ = self.forward(state)
            action = torch.tanh(mean) * self.max_action
            return action
        except Exception as e:
            print(f"获取确定性动作错误: {e}")
            return torch.zeros((state.shape[0], self.mean.out_features), device=state.device) * self.max_action


# 定义Critic网络（价值网络）
class CriticNetwork(nn.Module):
    """
    Critic网络：根据状态估计价值
    用于PPO算法中估计优势函数
    """
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # 添加层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        前向传播
        """
        # 检查输入是否包含NaN或Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        value = self.value(x)
        
        # 确保输出不含NaN
        value = torch.nan_to_num(value, nan=0.0)
        
        return value


# 定义PPO代理
class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 代理
    适合连续动作空间的强化学习算法
    """
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action=0.5,
        hidden_dim=128,
        lr_actor=1e-4,
        lr_critic=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        batch_size=128,
        update_epochs=5,
        buffer_size=1000
    ):
        # 初始化参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        
        # 初始化网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验回放缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []
        }
        self.buffer_size = buffer_size
        
    def select_action(self, state, training=True):
        """
        根据当前策略选择动作（带有错误处理）
        """
        try:
            # 检查输入状态是否有非法值
            if np.isnan(state).any() or np.isinf(state).any():
                # 替换非法值
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            if training:
                try:
                    action, log_prob = self.actor.sample(state)
                    return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]
                except Exception as e:
                    print(f"动作采样错误: {e}")
                    # 返回零动作和零对数概率作为后备
                    return np.zeros(self.action_dim), np.zeros(1)
            else:
                with torch.no_grad():
                    try:
                        action = self.actor.get_action(state)
                        return action.detach().cpu().numpy()[0], None
                    except Exception as e:
                        print(f"获取确定性动作错误: {e}")
                        return np.zeros(self.action_dim), None
        except Exception as e:
            print(f"选择动作错误: {e}")
            return np.zeros(self.action_dim), np.zeros(1) if training else None
            
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """
        存储转移到缓冲区
        """
        # 检查输入是否包含非法值
        if (np.isnan(state).any() or np.isinf(state).any() or 
            np.isnan(next_state).any() or np.isinf(next_state).any() or
            np.isnan(action).any() or np.isinf(action).any() or
            np.isnan(reward) or np.isinf(reward) or
            np.isnan(log_prob).any() or np.isinf(log_prob)):
            print("警告: 检测到非法值，跳过此转移")
            return
            
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        
        # 如果缓冲区已满，则清空
        if len(self.buffer['states']) >= self.buffer_size:
            self.update_policy()
            # 清空缓冲区
            for key in self.buffer.keys():
                self.buffer[key] = []
    
    def compute_gae(self, values, rewards, next_values, dones):
        """
        计算广义优势估计 (Generalized Advantage Estimation)
        """
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return advantages
            
    def update_policy(self):
        """
        使用PPO算法更新策略
        """
        if len(self.buffer['states']) < self.batch_size:
            print(f"警告: 缓冲区样本数量 ({len(self.buffer['states'])}) 小于批量大小 ({self.batch_size})，跳过更新")
            return
            
        try:
            # 将数据转换为张量
            states = torch.FloatTensor(np.array(self.buffer['states'])).to(device)
            actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(device)
            rewards = torch.FloatTensor(np.array(self.buffer['rewards'])).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(np.array(self.buffer['next_states'])).to(device)
            dones = torch.FloatTensor(np.array(self.buffer['dones'])).unsqueeze(1).to(device)
            old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(device)
            
            # 检查张量是否包含非法值
            for tensor, name in zip([states, actions, rewards, next_states, dones, old_log_probs], 
                               ["states", "actions", "rewards", "next_states", "dones", "old_log_probs"]):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"警告: {name} 张量包含NaN或Inf值，替换为合法值")
                    tensor = torch.nan_to_num(tensor)
            
            # 计算回报和优势
            with torch.no_grad():
                values = self.critic(states)
                next_values = self.critic(next_states)
                advantages = self.compute_gae(values, rewards, next_values, dones)
                advantages = torch.FloatTensor(advantages).to(device)
                returns = advantages + values
                
                # 标准化优势
                if len(advantages) > 1:  # 确保有足够的样本进行标准化
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 多次更新策略
            for _ in range(self.update_epochs):
                # 生成小批量索引
                indices = torch.randperm(states.size(0))
                for start in range(0, states.size(0), self.batch_size):
                    end = start + self.batch_size
                    if end > states.size(0):
                        end = states.size(0)
                    batch_indices = indices[start:end]
                    
                    # 获取小批量数据
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    try:
                        # 重新计算当前策略下的log概率
                        mean, std = self.actor(batch_states)
                        dist = Normal(mean, std)
                        
                        # 计算tanh的反函数，注意要处理边界情况
                        batch_actions_clipped = torch.clamp(batch_actions / self.max_action, -0.999, 0.999)
                        x = torch.atanh(batch_actions_clipped)
                        
                        new_log_probs = dist.log_prob(x)
                        new_log_probs -= torch.log(1 - batch_actions_clipped.pow(2) + 1e-6)
                        new_log_probs = new_log_probs.sum(1, keepdim=True)
                        
                        # 计算策略比率
                        ratio = torch.exp(torch.clamp(new_log_probs - batch_old_log_probs, -20, 20))
                        
                        # 计算PPO损失
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        # 更新Actor网络
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                        self.actor_optimizer.step()
                        
                        # 计算价值损失
                        value_preds = self.critic(batch_states)
                        value_loss = F.mse_loss(value_preds, batch_returns)
                        
                        # 更新Critic网络
                        self.critic_optimizer.zero_grad()
                        value_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                        self.critic_optimizer.step()
                    except Exception as e:
                        print(f"更新网络时出错: {e}")
                        continue
        except Exception as e:
            print(f"更新策略时出错: {e}")
    
    def save_models(self, path):
        """
        保存模型
        """
        try:
            os.makedirs(path, exist_ok=True)
            torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
            torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
            print(f"模型已保存到 {path}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
    
    def load_models(self, path):
        """
        加载模型
        """
        try:
            self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
            self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
            print(f"模型已从 {path} 加载")
        except Exception as e:
            print(f"加载模型时出错: {e}")


# 修改train函数，解决红点和灰点残留问题
def train(agent, env, creature_constructor, episodes=1000, max_steps=1000, render_interval=1, save_path="./models"):
    """
    训练代理
    参考rltest3实现的流畅动画方式，并解决点残留问题
    """
    os.makedirs(save_path, exist_ok=True)
    
    rewards_history = []
    avg_rewards_history = []
    best_avg_reward = float('-inf')
    
    # 创建状态归一化器
    state_normalizer = RunningMeanStd()
    
    # 创建奖励历史记录
    reward_history = deque(maxlen=100)
    
    # 设置地面高度
    ground_height = -50
    
    for episode in range(1, episodes + 1):
        try:
            # 完全重置屏幕和画布，清除所有点
            turtle.resetscreen()
            turtle.clearscreen()
            turtle.tracer(False)  # 关闭自动更新
            turtle.hideturtle()   # 隐藏海龟
            
            # 创建新的生物体和环境
            creature = creature_constructor()
            env = Environment([creature], g=100, dampk=0.5, groundhigh=ground_height)
            
            # 获取初始状态
            state_raw = creature.getstat(in3d=False, pk=0.01, vk=0.1, ak=0.001, conmid=True).numpy()
            
            # 检查状态合法性
            if np.isnan(state_raw).any() or np.isinf(state_raw).any():
                print(f"警告：回合 {episode} 的初始状态包含NaN或Inf")
                continue
            
            # 更新状态归一化器
            state_normalizer.update(np.array([state_raw]))
            state = state_normalizer.normalize(state_raw)
            
            episode_reward = 0
            initial_position = np.array([state_raw[-3], state_raw[-2], state_raw[-1]])
            
            # 准备物理环境可视化，在turtle设置完成后调用
            Phy.tready()
            
            for step in range(max_steps):
                # 选择动作
                action, log_prob = agent.select_action(state)
                
                # 检查动作是否合法并限制动作范围
                if np.isnan(action).any() or np.isinf(action).any():
                    action = np.zeros_like(action)
                action = np.clip(action, -0.5, 0.5)
                
                # 执行动作
                creature.act(action)
                env.step(0.01)
                
                # 获取下一状态和奖励
                next_state_raw = creature.getstat(in3d=False, pk=0.01, vk=0.1, ak=0.001, conmid=True).numpy()
                
                if np.isnan(next_state_raw).any() or np.isinf(next_state_raw).any():
                    print(f"警告：在回合 {episode} 步骤 {step} 中检测到非法下一状态")
                    break
                
                # 更新状态归一化器
                state_normalizer.update(np.array([next_state_raw]))
                next_state = state_normalizer.normalize(next_state_raw)
                
                # 计算重心位置和奖励
                position = next_state_raw[-3:]
                move_distance = min(position[0] - initial_position[0], 10)
                stability = -min(abs(position[1] - initial_position[1]) * 0.1, 5)
                energy_efficiency = -min(np.sum(np.square(action)) * 0.05, 1)
                reward = move_distance + stability + energy_efficiency
                reward = np.clip(reward, -10, 10)
                
                # 判断是否结束
                done = False
                if position[1] < ground_height:
                    done = True
                    reward -= 5
                
                # 存储转移
                agent.store_transition(state, action, reward, next_state, done, log_prob)
                
                state = next_state
                episode_reward += reward
                
                # 绘制地面线 - 使用白色线覆盖之前可能的残留点
                turtle.penup()
                turtle.goto(-800, ground_height-1)  # 稍微低一点以覆盖可能的点
                turtle.pendown()
                turtle.pencolor("white")  # 先用白色覆盖掉可能的残留点
                turtle.pensize(5)  # 稍微粗一些以覆盖点
                turtle.goto(800, ground_height-1)
                
                # 再绘制蓝色地面线
                turtle.penup()
                turtle.goto(-800, ground_height)
                turtle.pendown()
                turtle.pencolor("gray")
                turtle.pensize(2)
                turtle.goto(800, ground_height)
                turtle.penup()
                
                # 更新显示
                Phy.tplay()
                
                # 控制帧率
                time.sleep(0.01)
                
                if done:
                    break
                    
            # 记录回合奖励和其他统计信息
            reward_history.append(episode_reward)
            rewards_history.append(episode_reward)
            
            # 计算平均奖励
            filtered_rewards = [r for r in reward_history if -1000 < r < 1000]
            avg_reward = np.mean(filtered_rewards) if filtered_rewards else 0
            avg_rewards_history.append(avg_reward)
            
            # 打印训练信息
            print(f"回合 {episode}/{episodes}, 奖励: {episode_reward:.2f}, 平均奖励: {avg_reward:.2f}")
            
            # 保存表现更好的模型
            if avg_reward > best_avg_reward and episode > 10 and not np.isnan(avg_reward):
                best_avg_reward = avg_reward
                agent.save_models(save_path)
                print(f"模型保存于回合 {episode}, 平均奖励: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"回合 {episode} 出错: {e}")
            continue
    
    return rewards_history

# 修改evaluate函数，解决红点和灰点残留问题
def evaluate(agent, creature_constructor, num_episodes=5, max_steps=500):
    """
    评估代理性能
    参考rltest3实现的流畅动画方式，并解决点残留问题
    """
    rewards = []
    
    # 设置地面高度
    ground_height = -50
    
    for episode in range(num_episodes):
        try:
            # 完全重置屏幕和画布，清除所有点
            turtle.resetscreen()
            turtle.clearscreen()
            turtle.tracer(False)  # 关闭自动更新
            turtle.hideturtle()   # 隐藏海龟
            
            # 创建新环境和生物体
            creature = creature_constructor()
            env = Environment([creature], g=100, dampk=0.5, groundhigh=ground_height)
            
            state_raw = creature.getstat(in3d=False, pk=0.01, vk=0.1, ak=0.001, conmid=True).numpy()
            
            # 检查状态合法性
            if np.isnan(state_raw).any() or np.isinf(state_raw).any():
                print(f"警告：评估回合 {episode} 的初始状态包含NaN或Inf")
                continue
                
            state = state_raw  # 在评估时不使用归一化
            total_reward = 0
            initial_position = np.array([state_raw[-3], state_raw[-2], state_raw[-1]])
            
            # 准备物理环境可视化
            Phy.tready()
            
            for step in range(max_steps):
                # 选择确定性动作
                action, _ = agent.select_action(state, training=False)
                
                # 检查动作合法性
                if np.isnan(action).any() or np.isinf(action).any():
                    action = np.zeros_like(action)
                action = np.clip(action, -0.5, 0.5)
                
                # 执行动作
                creature.act(action)
                env.step(0.01)
                
                # 获取下一状态和奖励
                next_state_raw = creature.getstat(in3d=False, pk=0.01, vk=0.1, ak=0.001, conmid=True).numpy()
                
                if np.isnan(next_state_raw).any() or np.isinf(next_state_raw).any():
                    print(f"警告：在评估回合 {episode} 步骤 {step} 中检测到非法下一状态")
                    break
                    
                state = next_state_raw
                position = state[-3:]
                
                # 计算奖励
                move_distance = min(position[0] - initial_position[0], 10)
                stability = -min(abs(position[1] - initial_position[1]) * 0.1, 5)
                energy_efficiency = -min(np.sum(np.square(action)) * 0.05, 1)
                reward = move_distance + stability + energy_efficiency
                
                total_reward += reward
                
                # 绘制地面线
                turtle.penup()
                turtle.goto(-800, ground_height)
                turtle.pendown()
                turtle.pencolor("blue")
                turtle.pensize(3)
                turtle.goto(800, ground_height)
                turtle.penup()
                
                # 更新显示，直接使用Phy.tplay()
                Phy.tplay()
                
                # 控制帧率
                time.sleep(0.01)  # 约100FPS的渲染速度
                
                # 检查是否结束
                if position[1] < ground_height:
                    break
            
            rewards.append(total_reward)
            print(f"评估回合 {episode+1}/{num_episodes}, 奖励: {total_reward:.2f}")
            
        except Exception as e:
            print(f"评估回合 {episode} 出错: {e}")
            continue
    
    if rewards:
        print(f"平均评估奖励: {np.mean(rewards):.2f}")
    else:
        print("无有效评估回合")
    return rewards


# 主函数
def main():
    """
    主函数，执行训练和评估
    增加了对Turtle Graphics的支持
    """
    # 选择生物体类型
    creature_type = input("请选择生物体类型 (默认box2): ").strip() or "box2"
    
    # 获取生物体构造函数
    creature_constructors = {
        "box": box,
        "box2": box2,
        "box4": box4,
        "leg": leg,
        "leg2": leg2,
        "hat": hat,
        "insect": insect,
        "balance": balance,
        "balance2": balance2,
        "balance3": balance3,
        "humanb": humanb,
        "intrian": intrian
    }
    
    creature_constructor = creature_constructors.get(creature_type)
    if creature_constructor is None:
        raise ValueError(f"未知的生物体类型: {creature_type}")
    
    # 创建示例生物体以获取状态和动作维度
    creature = creature_constructor()
    state = creature.getstat(in3d=False, pk=0.01, vk=0.1, ak=0.001, conmid=True).numpy()
    state_dim = len(state)
    action_dim = len(creature.muscles)
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建代理（使用较小的学习率和更大的批量大小提高稳定性）
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=0.5,  # 减小最大动作值
        hidden_dim=128,
        lr_actor=1e-4,   # 降低学习率
        lr_critic=5e-4,  # 降低学习率
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        batch_size=128,   # 增大批量大小
        update_epochs=5,  # 减少更新次数
        buffer_size=1000  # 减小缓冲区大小
    )
    
    # 设置保存路径
    save_path = f"./models/{creature_type}"
    os.makedirs(save_path, exist_ok=True)
    
    # 训练模式
    train_mode = input("是否进行训练? (y/n): ").strip().lower() == 'y'
    
    if train_mode:
        # 训练
        episodes = int(input("请输入训练回合数 (默认500): ") or "500")
        rewards = train(
            agent=agent,
            env=None,  # Environment在训练函数中创建
            creature_constructor=creature_constructor,
            episodes=episodes,
            max_steps=500,  # 减少每回合步数
            render_interval=1,  # 每个回合都渲染
            save_path=save_path
        )
        
        # 绘制训练曲线
        try:
            # 配置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.title("训练奖励曲线")
            plt.xlabel("回合")
            plt.ylabel("奖励")
            plt.savefig(f"{save_path}/final_reward_plot.png")
            plt.show()
        except Exception as e:
            print(f"绘制奖励曲线时出错: {e}")
    else:
        # 加载已训练的模型
        try:
            agent.load_models(save_path)
        except Exception as e:
            print(f"无法加载模型: {e}")
            return
    
    # 评估
    evaluate(
        agent=agent,
        creature_constructor=creature_constructor,
        num_episodes=3,
        max_steps=500
    )


if __name__ == "__main__":
    main()