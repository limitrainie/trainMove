import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from physical import *
from myCreature import *
import os, time
import numpy as np

def mase(x, y):
    a = ((x - y) ** 2).sum()
    if a.item() < 1:
        return a
    else:
        return torch.sqrt(a)

def entropy(x):
    # 避免log(0)，添加小的epsilon值
    return -torch.sum(x * torch.log(x + 1e-10))

def clip(x, maxx, minx):
    return torch.clamp(x, min=minx, max=maxx)

def minten(x, y):
    return torch.min(x, y)

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

    def getstat(self):  # box21 leg35
        s = self.creatures[0].getstat(False, pk=0.023, vk=0.028, ak=0.001, mk=0.05)
        return torch.tensor(s, dtype=torch.float32)

    def act(self, a):
        self.creatures[0].actdisp(a)

    def reward(self):
        return self.r

    def show(self, m):
        e = env()
        Phy.tready()
        ar = 0
        # frame_time = 1/30  # 设置为60fps
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
            time.sleep(0.01)
        print(ar)

    def step(self, t):  # reward
        v = 0
        v2 = 0
        p = 0
        ang = 0
        std = 0.23  # 0.668930899
        mean = 0.069  # 0.103502928
        for i in range(30):
            super().step(t)
            # p+=sum([i.p[1] for i in self.creatures[0].phys])/len(self.creatures[0].phys)
            # v2+=sum([i.v[1] for i in self.creatures[0].phys])/len(self.creatures[0].phys)
            v += sum([i.v[0] for i in self.creatures[0].phys]) / len(self.creatures[0].phys)
            ang += ((self.plp[1].p[0] - self.plp[0].p[0]) * self.plumb[0] \
                  + (self.plp[1].p[1] - self.plp[0].p[1]) * self.plumb[1]) / distant(self.plp[0], self.plp[1])
            # p-=sum([1 if (i.x>=i.originx*i.maxl or i.x<=i.originx*i.minl) else 0 for i in self.creatures[0].muscles])*10
        # 奖励函数
        # 第1阶段奖励
        self.r = (v ** 0.5 / 90 if v > 1 else 0)  # 速度奖励
        self.r += -math.acos(ang / 30) / math.pi  # 姿态奖励
        self.r = (self.r - mean) / std / 3  # 标准化
        self.r -= 10 if self.isend(3) else 0  # 触地惩罚
        # self.r=max(0,v)/30#/120+0.05 
        # self.r=0
        
        # self.r+=-max(0,math.acos(ang/30)-math.acos(self.ang/30))/math.pi
        # self.r*=1-(math.acos(ang/30))/math.pi
        self.ang = ang
        # self.r-=(-v2)**0.5/90 if v2<0 else 0
        # self.r=(0.3 if v>1 else 0)-(10 if self.isend(3) else 0)
        # pos=sum([i.p[0] for i in self.creatures[0].phys])/len(self.creatures[0].phys)
        # self.r=0
        # if pos>self.flag:
        #     self.r=(pos-self.flag)
        #     self.flag=pos

        # r9=max(0,v)/30
        # (self.r-mean=20.3)/std=30.9/3

        # r10=max(0,v)/30/120+0.05

        # print(self.r)
    
    def test(self, times=10):
        for t in range(times):
            e = env()
            ar = 0
            for i in range(n):
                e.act([random.randint(0, 1) for i in range(musclenum)])  # [0,1] if e.creatures[0].phys[3].p[0]<0 else [1,0]
                e.step(0.001)
                ar += e.reward()
                p = 0
                v = 0
                a = 0
                m = 0
                for i in e.creatures[0].phys:
                    p += (i.p[0] + i.p[1]) / 2
                    v += (i.v[0] + i.v[1]) / 2
                    a += (i.axianshi[0] + i.axianshi[1]) / 2
                for i in e.creatures[0].muscles:
                    m += distant(i.p1, i.p2)
                p /= len(e.creatures[0].phys)
                v /= len(e.creatures[0].phys)
                a /= len(e.creatures[0].phys)
                m /= len(e.creatures[0].muscles)
                print(e.reward(), p, v, a, m)
                if e.isend():
                    break
    
    def isend(self, h=1):  
        for i in self.creatures[0].phys:
            if i not in self.foot and i.p[1] < h + self.ground:
                # 如果身体着地停止训练
                return True
        return False

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.f1 = nn.Linear(statnum, 30)
        self.fh = nn.ModuleList([nn.Linear(30, 30) for _ in range(6)])
        self.f2 = nn.Linear(30, musclenum * 2)

    def forward(self, x):
        x = F.relu(self.f1(x))
        for layer in self.fh:
            x = x + F.relu(layer(x))
        x = self.f2(x)
        
        # 分割并应用softmax到每对输出
        result = []
        for i in range(musclenum):
            pair = x[i*2:i*2+2]
            soft_pair = F.softmax(pair, dim=0)
            result.append(soft_pair)
        
        return torch.cat(result)

    def choice(self, x):
        with torch.no_grad():
            v = self.forward(x).numpy()
            a = []
            
            for i in range(0, len(v), 2):
                v2 = v[i:i+2]
                chosen = np.random.choice([0, 1], p=v2)
                a.append(chosen)
                
            return a

class modelv(nn.Module):
    def __init__(self):
        super(modelv, self).__init__()
        self.f1 = nn.Linear(statnum, 30)
        self.fh = nn.ModuleList([nn.Linear(30, 30) for _ in range(6)])
        self.f2 = nn.Linear(30, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        for layer in self.fh:
            x = x + F.relu(layer(x))
        x = self.f2(x)
        return x

class memory:
    def __init__(self, maxsize=10):
        self.memo = []
        self.maxsize = maxsize
    
    def experience(self, m, times=3, n=500):
        for t in range(times):
            e = env()
            exp = []
            ar = 0
            for i in range(n):
                s = e.getstat()
                with torch.no_grad():
                    v = m.forward(s).numpy()
                a = []
                p = []
                for j in range(0, len(v), 2):
                    v2 = v[j:j+2]
                    chosen = np.random.choice([0, 1], p=v2)
                    a.append(chosen)
                    p.append(v2[chosen])
                
                e.act(a)
                e.step(0.001)
                st = e.getstat()
                r = e.reward()
                ar += r
                exp.append([s, a, r, st, 0, p])
                if i == n - 1 or e.isend():
                    exp[-1][4] = 1
                    break
            self.memo.append(exp)
        if len(self.memo) > self.maxsize:
            self.memo = self.memo[-self.maxsize:]
        return ar / times

def train(m, mv, memo, n=200, times=1, discount=0.99, lamb=0.99, ek=0.5, eps=0.2):
    t0 = time.perf_counter()
    ar = memo.experience(m, times, n=n)
    t1 = time.perf_counter()
    
    m_optimizer = optim.Adam(m.parameters(), lr=0.0008)
    mv_optimizer = optim.Adam(mv.parameters(), lr=0.0008)
    
    aloss = 0
    alossv = 0
    alosse = 0
    alratio = 0
    count = 0
    
    for exp in memo.memo:
        c = 0
        ad = []
        gae = 0
        
        # 计算GAE
        for i in range(len(exp) - 1, -1, -1):
            s = exp[i][0]
            st = exp[i][3]
            
            with torch.no_grad():
                v = mv(s).item()
                v2 = mv(st).item()
                
            tdd = exp[i][2] + discount * v2 * (1 - exp[i][4]) - v
            gae = tdd + lamb * discount * gae * (1 - exp[i][4])
            ad.append(gae)
            
        ad.reverse()
        
        for i in exp:
            s, a, r, st, end, p = i
            
            # 模型前向传播
            out = m(s)
            pc = torch.tensor([out[a[j]*2 + j].item() for j in range(len(a))], dtype=torch.float32)
            
            ent = entropy(out)
            v = mv(s)
            adv = torch.tensor(ad[c], dtype=torch.float32)
            
            old_probs = torch.tensor(p, dtype=torch.float32)
            ratio = torch.exp(torch.log(pc) - torch.log(old_probs))
            
            surr1 = ratio * adv
            surr2 = clip(ratio, 1 + eps, 1 - eps) * adv
            
            # PPO损失函数
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -ek * ent
            critic_loss = mase(v, torch.tensor([adv.item()]))
            
            # 总损失
            loss = actor_loss + entropy_loss + critic_loss
            
            # 优化步骤
            m_optimizer.zero_grad()
            mv_optimizer.zero_grad()
            loss.backward()
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
    print(ar, aloss/count, alossv/count, alosse/count, alratio/count, t1-t0, t2-t1)
    return ar, aloss/count, alossv/count, alosse/count

# 全局设置
evnname = "box2"
lastname = "-deep-r11"
e = env()
statnum = len(e.getstat())
musclenum = sum([len(i.muscles) for i in e.creatures])
ground = e.ground
del e

savename = f"rlt-3-ppo-{evnname}{lastname}"
if savename in os.listdir():
    print("load", savename)
    checkpoint = torch.load(savename)
    m = model()
    mv = modelv()
    m.load_state_dict(checkpoint['model_state_dict'])
    mv.load_state_dict(checkpoint['modelv_state_dict'])
else:
    m = model()
    mv = modelv()

memo = memory(8)
n = 500   # 一轮训练的最大回合数

mode = 1  # =1训练，=0测试模型
ek = 0.5  # # 熵系数
ae = 0.3

while True:
    if mode:
        try:
            # ek = min(max(ek * 2 if ae < 0.2 else ek / 2 if ae > 0.33 else ek, 0.1), 3)
            ek = 2 if ae < 0.2 else 0.5
            r, al, av, ae = train(m, mv, memo, discount=0.98, ek=ek, n=n)
            
            # 保存模型
            torch.save({
                'model_state_dict': m.state_dict(),
                'modelv_state_dict': mv.state_dict(),
            }, savename)
            
        except OverflowError:
            r = 0
            print("OverflowError")
            memo.memo = []
            
        except ZeroDivisionError:
            r = 0
            print("ZeroDivisionError")
            memo.memo = []
            
    else:
        e = env()
        e.show(m)

# e = env()
# e.test()