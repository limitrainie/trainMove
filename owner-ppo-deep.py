import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from physical import *
from myCreature import *
import os, time, random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# 训练监控器类
class TrainingMonitor:
    """训练监控器：记录和可视化PPO训练过程"""
    def __init__(self, save_dir="training_logs"):
        # 设置matplotlib中文字体
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.family'] = 'sans-serif'

        # 创建保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化数据收集列表
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        self.policy_ratios = []
        self.experience_times = []
        self.update_times = []
        self.iterations = []
        self.current_iter = 0
        
        # 记录训练开始时间
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(save_dir, f"training_log_{self.start_time}.txt")
        
        # 创建日志文件标题
        with open(self.log_file, "w") as f:
            f.write("Iteration,Reward,ActorLoss,CriticLoss,Entropy,PolicyRatio,ExperienceTime,UpdateTime\n")
    
    def log_iteration(self, reward, actor_loss, critic_loss, entropy, policy_ratio, exp_time, update_time):
        """记录单次迭代的训练数据"""
        # 添加数据到列表
        self.current_iter += 1
        self.iterations.append(self.current_iter)
        self.rewards.append(reward)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.entropy_values.append(entropy)
        self.policy_ratios.append(policy_ratio)
        self.experience_times.append(exp_time)
        self.update_times.append(update_time)
        
        # 写入日志文件
        with open(self.log_file, "a") as f:
            f.write(f"{self.current_iter},{reward},{actor_loss},{critic_loss},{entropy},{policy_ratio},{exp_time},{update_time}\n")
        
        # 定期保存图表
        if self.current_iter % 20 == 0:
            self.save_all_plots()
            self.save_data()
    
    def save_data(self):
        """保存收集的训练数据"""
        np.savez(os.path.join(self.save_dir, f"training_data_{self.start_time}.npz"),
                 iterations=np.array(self.iterations),
                 rewards=np.array(self.rewards),
                 actor_losses=np.array(self.actor_losses),
                 critic_losses=np.array(self.critic_losses),
                 entropy_values=np.array(self.entropy_values),
                 policy_ratios=np.array(self.policy_ratios),
                 experience_times=np.array(self.experience_times),
                 update_times=np.array(self.update_times))
    
    def plot_reward_curve(self):
        """绘制奖励曲线图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.rewards, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # 添加移动平均线
        if len(self.rewards) > 10:
            window_size = min(10, len(self.rewards)//4)
            rewards_smooth = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(self.iterations[window_size-1:], rewards_smooth, 'r-', linewidth=1.5, alpha=0.8, label=f'移动平均 ({window_size})')
            plt.legend()
        
        # 填充区域
        plt.fill_between(self.iterations, self.rewards, 0, 
                         where=(np.array(self.rewards) > 0), 
                         color='skyblue', alpha=0.3, interpolate=True)
        plt.fill_between(self.iterations, self.rewards, 0, 
                         where=(np.array(self.rewards) < 0), 
                         color='salmon', alpha=0.3, interpolate=True)
        
        plt.title('训练奖励随时间变化曲线', fontsize=16)
        plt.xlabel('训练迭代次数', fontsize=14)
        plt.ylabel('平均奖励', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_losses(self):
        """绘制各种损失函数曲线图"""
        plt.figure(figsize=(12, 7))
        plt.plot(self.iterations, self.actor_losses, 'r-', label='Actor Loss', linewidth=2)
        plt.plot(self.iterations, self.critic_losses, 'g-', label='Critic Loss', linewidth=2)
        plt.plot(self.iterations, self.entropy_values, 'b-', label='Entropy', linewidth=2)
        
        plt.title('训练损失值变化曲线', fontsize=16)
        plt.xlabel('训练迭代次数', fontsize=14)
        plt.ylabel('损失值', fontsize=14)
        plt.legend(fontsize=12)
        
        # 根据数据范围选择是否使用对数尺度
        if len(self.actor_losses) > 0 and max(self.actor_losses) / (min(self.actor_losses) + 1e-10) > 100:
            plt.yscale('log')
            plt.title('训练损失值变化曲线 (对数尺度)', fontsize=16)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_entropy_policy_ratio(self):
        """绘制熵和策略比率曲线图"""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('训练迭代次数', fontsize=14)
        ax1.set_ylabel('熵值', color=color, fontsize=14)
        ax1.plot(self.iterations, self.entropy_values, color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 添加0.2参考线
        ax1.axhline(y=0.2, color='b', linestyle='--', alpha=0.6, label='熵阈值 (0.2)')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('策略比率偏差', color=color, fontsize=14)
        ax2.plot(self.iterations, self.policy_ratios, color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 添加0.2参考线
        ax2.axhline(y=0.2, color='r', linestyle='--', alpha=0.6, label='策略比率裁剪 (0.2)')
        
        # 添加两个图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('探索程度(熵)和策略稳定性变化', fontsize=16)
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig
    
    def plot_training_time(self):
        """绘制训练时间分析图"""
        plt.figure(figsize=(12, 6))
        
        bar_width = 0.8
        
        plt.bar(self.iterations, self.experience_times, bar_width, 
                label='经验收集时间', color='skyblue')
        plt.bar(self.iterations, self.update_times, bar_width, 
                bottom=self.experience_times, label='策略更新时间', color='salmon')
        
        plt.title('每次迭代的训练时间分布', fontsize=16)
        plt.xlabel('训练迭代次数', fontsize=14)
        plt.ylabel('时间 (秒)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_all_plots(self):
        """保存所有图表"""
        # 创建本次保存的目录
        plots_dir = os.path.join(self.save_dir, f"plots_{self.start_time}")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 保存各个图表
        fig1 = self.plot_reward_curve()
        fig1.savefig(os.path.join(plots_dir, "reward_curve.png"), dpi=300)
        plt.close(fig1)
        
        fig2 = self.plot_losses()
        fig2.savefig(os.path.join(plots_dir, "losses.png"), dpi=300)
        plt.close(fig2)
        
        fig3 = self.plot_entropy_policy_ratio()
        fig3.savefig(os.path.join(plots_dir, "entropy_policy_ratio.png"), dpi=300)
        plt.close(fig3)
        
        fig4 = self.plot_training_time()
        fig4.savefig(os.path.join(plots_dir, "training_time.png"), dpi=300)
        plt.close(fig4)
        
        print(f"图表已保存到 {plots_dir}")

# 辅助函数
def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-10))

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """计算广义优势估计(GAE)"""
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
    return advantages

# 环境类
class env(Environment):
    def __init__(self):
        Phy.biao = []
        Phy.rbiao = []
        super().__init__([eval(evnname + "()")],
                         g=650,
                         groundhigh=-50,
                         groundk=10000,
                         grounddamp=100,
                         randsigma=0.1,
                         dampk=0.08,
                         friction=650)
        for i in self.creatures[0].muscles:
            i.stride = 3
            i.damk = 20
        self.r = 0
        if len(self.creatures[0].skeletons) != 0:
            self.plp = [self.creatures[0].skeletons[0].p1, self.creatures[0].skeletons[0].p2]
        else:
            self.plp = [self.creatures[0].phys[0], self.creatures[0].phys[1]]
        self.plumb = [(self.plp[1].p[0] - self.plp[0].p[0]) / distant(self.plp[0], self.plp[1]),
                      (self.plp[1].p[1] - self.plp[0].p[1]) / distant(self.plp[0], self.plp[1])]
        self.ang = 0
        self.foot = [i for i in self.creatures[0].phys if i.p[1] <= 0]
        self.flag = sum([i.p[0] for i in self.creatures[0].phys]) / len(self.creatures[0].phys)  # 初始化flag为初始质心x坐标
        self.last_pos = self.flag  # 记录上一步位置

    def getstat(self):
        s = self.creatures[0].getstat(False, pk=0.023, vk=0.028, ak=0.001, mk=0.05)
        # 使用clone().detach()代替torch.tensor()
        if isinstance(s, torch.Tensor):
            return s.clone().detach().to(dtype=torch.float32)
        else:
            return torch.tensor(s, dtype=torch.float32)

    def act(self, a):
        self.creatures[0].actdisp(a)

    def reward(self):
        return self.r

    def show(self, m):
        e = env()
        Phy.tready()
        ar = 0
        for i in range(n):
            a = m.choice(e.getstat())
            e.act(a)
            e.step(0.001)
            ar += e.reward()
            turtle.goto(-800, ground)
            turtle.pendown()
            turtle.goto(800, ground)
            turtle.penup()
            Phy.tplay()
            if e.isend():
                break
            time.sleep(0.001)
        print(f"总奖励: {ar:.4f}")

    def step(self, t):
        # 调用父类的物理模拟
        super().step(t)
        
        # 计算当前质心 x 坐标
        pos = sum(p.p[0] for p in self.creatures[0].phys) / len(self.creatures[0].phys)
        # 向右移动增量
        delta = pos - self.last_pos
        # 奖励 = 向右移动量（可以乘以系数放大）
        self.r = delta * 2.0
        
        # 对self.r进行标准化，确保在[0, 1]范围内
        self.r = 1.0 / (1.0 + math.exp(-self.r))
        
        # 如果倒地，给一次性惩罚
        if self.isend():
            self.r -= 1.0
        
        # 更新 last_pos
        self.last_pos = pos
    
    def test(self, times=10):
        for t in range(times):
            e = env()
            ar = 0
            for i in range(n):
                e.act([random.randint(0, 1) for i in range(musclenum)])
                e.step(0.001)
                ar += e.reward()
                if e.isend():
                    break
            print(f"测试回合 {t+1}: 奖励 {ar:.4f}")
    
    def isend(self, h=1):  
        for i in self.creatures[0].phys:
            if i not in self.foot and i.p[1] < h + self.ground:
                return True
        return False
    
    def reset(self):
        """重置环境状态，返回初始状态"""
        # 重新初始化环境
        self.__init__()
        return self.getstat()

# 模型定义 - 优化的Actor网络
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(statnum, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Linear(64, musclenum * 2)
        # )
        self.fc = nn.Sequential(
            nn.Linear(statnum, 128),  # 提高层宽度
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, musclenum * 2)
        )
        
    def forward(self, x):
        # 检查是否有batch维度，没有则添加
        need_squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加batch维度
            need_squeeze = True
            
        # 前向传播
        logits = self.fc(x)  # [batch_size, musclenum*2]
        
        batch_size = logits.shape[0]
        muscle_count = logits.shape[1] // 2  # 计算肌肉数量
        
        # 重塑为[batch_size, musclenum, 2]便于应用softmax
        logits = logits.view(batch_size, muscle_count, 2)
        
        # 对每个肌肉的动作对应用softmax
        probs = F.softmax(logits, dim=2)
        
        # 如果是单样本输入，去掉添加的batch维度
        if need_squeeze:
            probs = probs.squeeze(0)
            
        return probs
    
    def choice(self, x):
        with torch.no_grad():
            probs = self.forward(x).reshape(-1, 2)
            actions = []
            for prob in probs:
                action = torch.multinomial(prob, 1).item()
                actions.append(action)
            return actions
    
    def log_prob(self, out, actions):
        """计算动作的对数概率"""
        out_reshaped = out.reshape(-1, 2)
        actions_tensor = torch.tensor(actions)
        log_probs = []
        
        for i, action in enumerate(actions_tensor):
            log_probs.append(torch.log(out_reshaped[i, action] + 1e-10))
            
        return torch.stack(log_probs)
        
    def entropy(self, out):
        """计算策略的熵"""
        out_reshaped = out.reshape(-1, 2)
        entropies = []
        
        for probs in out_reshaped:
            ent = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append(ent)
            
        return torch.stack(entropies)

# 优化的价值网络
class modelv(nn.Module):
    def __init__(self):
        super(modelv, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(statnum, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# 批量轨迹收集
def collect_trajectories(policy_net, env_fn, n_envs=4, max_steps=200):
    """并行收集多个环境的轨迹"""
    # 初始化N个环境
    envs = [env_fn() for _ in range(n_envs)]
    # 存储所有环境的轨迹
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    
    # 初始化状态
    states = [e.getstat() for e in envs]
    dones = [False for _ in range(n_envs)]
    
    # 收集轨迹
    for _ in range(max_steps):
        # 为每个环境采取动作
        batch_states = torch.stack([s for i, s in enumerate(states) if not dones[i]])
        with torch.no_grad():
            batch_probs = policy_net(batch_states)
        
        actions = []
        log_probs = []
        rewards = []
        next_states = []
        new_dones = []
        
        active_idx = 0
        for i in range(n_envs):
            if dones[i]:
                # 已完成的环境跳过
                actions.append(None)
                log_probs.append(None)
                rewards.append(0)
                next_states.append(states[i])
                new_dones.append(True)
                continue
            
            # 获取当前环境的动作概率
            probs = batch_probs[active_idx:active_idx+1]
            active_idx += 1
            
            # 选择动作
            action = policy_net.choice(states[i])
            log_prob = policy_net.log_prob(probs, action)
            
            # 执行动作
            envs[i].act(action)
            envs[i].step(0.001)
            reward = envs[i].reward()
            next_state = envs[i].getstat()
            done = envs[i].isend() or _ == max_steps - 1
            
            # 存储结果
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            new_dones.append(done)
        
        # 更新状态和完成标志
        all_states.append(states)
        all_actions.append(actions)
        all_log_probs.append(log_probs)
        all_rewards.append(rewards)
        all_dones.append(dones)
        
        states = next_states
        dones = new_dones
        
        # 如果所有环境都完成，提前退出
        if all(dones):
            break
    
    return all_states, all_actions, all_rewards, all_dones, all_log_probs

# 动态调整的PPO训练过程
def ppo_update(policy_net, value_net, optimizer_p, optimizer_v, 
              states, actions, old_log_probs, rewards, dones, 
              gamma=0.99, lam=0.95, eps=0.2, epochs=4, batch_size=64):
    """使用PPO算法更新策略和价值网络"""
    # 扁平化并过滤掉done的数据
    flat_states = []
    flat_actions = []
    flat_old_log_probs = []
    flat_rewards = []
    flat_dones = []
    
    # 计算GAE和目标值
    with torch.no_grad():
        values = [value_net(s).squeeze() for s in states]
    
    advantages = []
    returns = []
    
    next_value = 0
    gae = 0
    
    for t in reversed(range(len(states))):
        masks = [1.0 - float(done) for done in dones[t]]
        next_value = next_value * masks[0]  # 简化，假设所有环境同步
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * masks[0] * gae
        
        next_value = values[t]
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    # 转换为张量并标准化advantages
    advantages = torch.cat(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.cat(returns)
    
    # 构建数据集
    flat_states = torch.cat([s for s in states if s is not None], dim=0)
    flat_actions = [a for acts in actions for a in acts if a is not None]
    flat_log_probs = torch.cat([lp for lp in old_log_probs if lp is not None], dim=0)
    
    dataset = TensorDataset(flat_states, flat_actions, flat_log_probs, advantages, returns)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 多次更新
    for _ in range(epochs):
        for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
            # 计算新的动作概率
            probs = policy_net(batch_states)
            new_log_probs = policy_net.log_prob(probs, batch_actions)
            entropy = policy_net.entropy(probs).mean()
            
            # 计算比率并裁剪
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1-eps, 1+eps) * batch_advantages
            
            # 计算actor和critic的损失
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.smooth_l1_loss(value_net(batch_states).squeeze(), batch_returns)
            
            # 添加熵奖励
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # 更新网络
            optimizer_p.zero_grad()
            optimizer_v.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            
            optimizer_p.step()
            optimizer_v.step()

# 优化后的记忆类
class memory:
    def __init__(self, maxsize=48):  # 增大记忆容量
        self.memo = []
        self.maxsize = maxsize
    
    def experience(self, m, times=5, n=500):  # 增加采样次数
        total_rewards = 0
        
        for _ in range(times):
            e = env()  # 创建新环境
            exp = []  # 创建空的经验列表
            ep_reward = 0  # 初始化回合奖励
            
            for i in range(n):
                s = e.getstat()
                with torch.no_grad():
                    probs = m.forward(s)
                    # 确保是二维张量[musclenum, 2]
                    if probs.dim() == 3:  # 如果是[1, musclenum, 2]
                        probs = probs.squeeze(0)
                    v = probs.cpu().numpy() if probs.is_cuda else probs.numpy()
                        
                a = []
                p = []
                for j in range(musclenum):
                    v2 = v[j]  # 获取第j个肌肉的两个动作概率
                    chosen = np.random.choice([0, 1], p=v2)
                    a.append(chosen)
                    p.append(v2[chosen])
                
                e.act(a)
                e.step(0.001)
                st = e.getstat()
                r = e.reward()
                ep_reward += r
                exp.append([s, a, r, st, 0, p])
                if i == n - 1 or e.isend():
                    exp[-1][4] = 1
                    break
                
            self.memo.append(exp)
            total_rewards += ep_reward
        
        # 保持记忆大小有限
        if len(self.memo) > self.maxsize:
            self.memo = self.memo[-self.maxsize:]
        
        return total_rewards / times

# 优化后的训练函数
def train(m, mv, memo, n=200, times=1, discount=0.99, lamb=0.95, ek=0.5, eps=0.2):
    # 获取当前训练迭代次数和动态调整学习率
    current_iter = monitor.current_iter + 1
    
    # 动态学习率调度
    base_actor_lr = 3e-4
    base_critic_lr = 1e-3
    base_actor_lr = 1e-3  # 提高初始值
    actor_lr = base_actor_lr * (0.9998 ** current_iter)
    critic_lr = base_critic_lr * (0.9995 ** current_iter)
    
    # 动态调整裁剪参数
    clip_eps = 0.3 - 0.2 * min(1.0, current_iter / 1000)  # 从0.3线性衰减到0.1
    
    # 动态调整熵系数
    entropy_coef = max(0.01, ek * (0.98 ** current_iter)) if current_iter < 300 else 0.001
    
    # 优化器
    m_optimizer = optim.Adam(m.parameters(), lr=actor_lr)
    mv_optimizer = optim.Adam(mv.parameters(), lr=critic_lr)
    
    # 记录时间
    t0 = time.perf_counter()
    ar = memo.experience(m, times, n=n)
    t1 = time.perf_counter()
    
    # 统计变量
    aloss = 0
    alossv = 0
    alosse = 0
    alratio = 0
    count = 0
    
    # 批量更新网络
    for exp in memo.memo:
        c = 0
        # 计算GAE
        ad = []
        returns = []  # 添加returns列表
        gae = 0
        
        # 从后往前计算GAE
        for i in range(len(exp) - 1, -1, -1):
            s = exp[i][0]
            st = exp[i][3]
            
            with torch.no_grad():
                v = mv(s).item()
                v2 = mv(st).item()
                
            # TD残差
            tdd = exp[i][2] + discount * v2 * (1 - exp[i][4]) - v
            # GAE计算
            gae = tdd + lamb * discount * gae * (1 - exp[i][4])
            ad.append(gae)
            # 计算returns (累积折现奖励)
            returns.append(gae + v)  # 回报 = 优势 + 价值估计
        
        ad.reverse()
        returns.reverse()  # 因为是从后往前计算，所以需要反转列表
        
        # 计算优势标准化
        adv_tensor = torch.tensor(ad, dtype=torch.float32)
        adv_normalized = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)  # 转换为张量
        
        # 对每个时间步的经验进行批量更新
        for batch_start in range(0, len(exp), 32):
            batch_indices = range(batch_start, min(batch_start + 32, len(exp)))
            batch_states = torch.stack([exp[i][0] for i in batch_indices])
            batch_actions = [exp[i][1] for i in batch_indices]
            batch_probs = [exp[i][5] for i in batch_indices]
            batch_adv = adv_normalized[batch_indices]
            batch_returns = returns_tensor[batch_indices]  # 获取批次的returns
            
            # 前向传播
            out = m(batch_states)  # 形状: [batch_size, musclenum, 2]
            pc_list = []
            for j, actions in enumerate(batch_actions):
                # 提取该样本的所有肌肉动作概率
                sample_probs = out[j]  # 形状: [musclenum, 2]
                # 收集每个肌肉选择的动作概率
                selected_probs = torch.tensor([sample_probs[k, action] 
                                              for k, action in enumerate(actions)], 
                                              dtype=torch.float32)
                pc_list.append(selected_probs)
            
            pc = torch.cat(pc_list)
            ent = entropy(out)
            v = mv(batch_states)
            old_probs = torch.tensor([prob for probs in batch_probs for prob in probs], dtype=torch.float32)
            ratio = torch.exp(torch.log(pc) - torch.log(old_probs))  # 现在old_probs已定义
            # 获取每个样本的肌肉数量
            muscles_per_sample = len(batch_actions[0])
            # 扩展batch_adv以匹配pc和old_probs的维度
            expanded_adv = torch.repeat_interleave(batch_adv, muscles_per_sample)
            # 使用扩展后的优势
            surr1 = ratio * expanded_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * expanded_adv
            
            # 损失函数
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy_coef * ent
            v_pred = v.squeeze(-1)  # 从[batch_size,1]变为[batch_size]
            critic_target = batch_returns  # 使用真实回报
            critic_loss = F.smooth_l1_loss(v_pred, critic_target)
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            # 优化步骤
            m_optimizer.zero_grad()
            mv_optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(mv.parameters(), max_norm=0.5)
            
            m_optimizer.step()
            mv_optimizer.step()
            
            # 统计数据
            aloss += actor_loss.item()
            alosse += ent.item() / len(out)
            alratio += torch.mean(torch.abs(ratio - 1)).item()
            alossv += critic_loss.item()
            
            count += 1
            c += 1
    
    t2 = time.perf_counter()
    print(f"奖励:{ar:.4f} 策略损失:{aloss/count:.4f} 价值损失:{alossv/count:.4f} "
          f"熵:{alosse/count:.4f} 比率:{alratio/count:.6f} "
          f"采样:{t1-t0:.2f}s 更新:{t2-t1:.2f}s")
    
    # 记录训练数据
    monitor.log_iteration(ar, aloss/count, alossv/count, alosse/count, alratio/count, t1-t0, t2-t1)
    
    return ar, aloss/count, alossv/count, alosse/count

# 主程序
if __name__ == "__main__":
    # 全局设置
    evnname = "box1"
    lastname = "-deep-optimized-v6"
    e = env()
    statnum = len(e.getstat())
    musclenum = sum([len(i.muscles) for i in e.creatures])
    ground = e.ground
    del e
    
    savename = f"models/rlt-3-ppo-{evnname}{lastname}"
    best_model_name = f"{savename}_best"
    
    if savename in os.listdir():
        print(f"加载模型: {best_model_name}")
        checkpoint = torch.load(best_model_name)
        m = model()
        mv = modelv()
        m.load_state_dict(checkpoint['model_state_dict'])
        mv.load_state_dict(checkpoint['modelv_state_dict'])
    else:
        print("创建新模型")
        m = model()
        mv = modelv()
    
    memo = memory(24)  # 增大记忆容量
    n = 5000  # 一轮训练的最大回合数
    
    # 初始化训练监控器
    monitor = TrainingMonitor()
    
    mode = 0  # =1训练，=0测试模型
    best_reward = float('-inf')
    no_improvement_count = 0
    
    try:
        while True:
            if mode:
                try:
                    # 动态调整熵系数
                    ek = min(max(1.0 + (0.2 - ae if 'ae' in locals() else 0.3) * 4.0, 0.3), 2.5)
                    
                    r, al, av, ae = train(m, mv, memo, times=8 , discount=0.98, ek=ek, n=n)
                    
                    # 检查是否有改进
                    if r > best_reward:
                        best_reward = r
                        no_improvement_count = 0
                        print(f"发现更好的模型，奖励: {r:.2f}，保存为 {best_model_name}")
                        # 保存最佳模型
                        torch.save({
                            'model_state_dict': m.state_dict(),
                            'modelv_state_dict': mv.state_dict(),
                            'iteration': monitor.current_iter,
                            'best_reward': r
                        }, best_model_name)
                    else:
                        no_improvement_count += 1
                        
                    # 定期保存当前模型
                    torch.save({
                        'model_state_dict': m.state_dict(),
                        'modelv_state_dict': mv.state_dict(),
                        'iteration': monitor.current_iter
                    }, savename)
                    
                    # 如果长时间没有进步，恢复到最佳模型
                    if no_improvement_count >= 50:
                        print(f"50次迭代无改善，恢复到最佳模型")
                        if os.path.exists(best_model_name):
                            checkpoint = torch.load(best_model_name)
                            m.load_state_dict(checkpoint['model_state_dict'])
                            mv.load_state_dict(checkpoint['modelv_state_dict'])
                            print(f"已恢复到最佳模型，奖励: {checkpoint.get('best_reward', 'unknown')}")
                        no_improvement_count = 0
                        
                except OverflowError:
                    print("数值溢出错误，清空记忆并继续")
                    memo.memo = []
                    
                except ZeroDivisionError:
                    print("除零错误，清空记忆并继续")
                    memo.memo = []
                    
            else:
                e = env()
                e.show(m)
                
    except KeyboardInterrupt:
        print("训练中断，保存数据...")
        monitor.save_data()
        monitor.save_all_plots()
        print("数据已保存，程序退出。")