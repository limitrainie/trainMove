# -*- coding: utf-8 -*-
"""
盒子物理模型 - 基于优化版物理引擎
"""

import torch
import random
from typing import List, Optional
import math
import time
import turtle

# 从优化版本物理引擎导入
from physical import Phy, DingPhy


def distant(p1, p2):
    """计算两点之间的距离，添加安全检查防止溢出"""
    try:
        # 安全计算坐标差
        dx = p1.p[0] - p2.p[0]
        dy = p1.p[1] - p2.p[1]
        dz = p1.p[2] - p2.p[2]
        
        # 检查坐标差是否过大
        if abs(dx) > 1e6 or abs(dy) > 1e6 or abs(dz) > 1e6:
            # 避免精确计算，直接返回一个足够大的值
            return 1e8
        
        # 逐步计算平方和，避免中间结果溢出
        dist_squared = 0.0
        dist_squared += dx * dx
        dist_squared += dy * dy
        dist_squared += dz * dz
        
        # 安全计算平方根
        return math.sqrt(dist_squared)
    except OverflowError:
        # 如果仍然溢出，返回一个大但安全的值
        print("警告: 距离计算溢出")
        return 1e8
    except Exception as e:
        print(f"距离计算错误: {e}")
        return 1e8


def damp(p, k):
    """应用阻尼力，添加安全检查"""
    try:
        # 限制速度防止溢出
        vx = max(min(p.v[0], 1e6), -1e6)
        vy = max(min(p.v[1], 1e6), -1e6)
        vz = max(min(p.v[2], 1e6), -1e6)
        
        # 计算并应用阻尼力
        p.force([-k * vx, -k * vy, -k * vz])
    except Exception as e:
        print(f"应用阻尼错误: {e}")


class Muscle:
    """肌肉类：控制两个物理点之间的收缩和舒张"""
    
    def __init__(self, p1, p2, x = None, k = 1000, maxl = 1.5, minl = 0.1, stride = 2, dampk = 20):
        """
        创建肌肉
        
        参数:
            p1, p2: 连接的两个物理点
            x: 肌肉原长
            k: 弹簧劲度系数
            maxl, minl: 肌肉最大/最小长度（相对于原长的比例）
            stride: 每次动作的步长
            dampk: 阻尼系数
        """
        self.p1 = p1
        self.p2 = p2
        
        # 安全计算初始长度
        try:
            self.x = distant(p1, p2) if x is None else x
        except:
            print("肌肉初始化出错，使用默认长度100")
            self.x = 100.0
            
        self.originx = self.x
        self.k = k
        self.dampk = dampk
        self.minl = minl
        self.maxl = maxl
        self.stride = stride
    
    def regulation(self):
        """调整肌肉长度在允许范围内"""
        self.x = max(self.x, self.originx * self.minl)
        self.x = min(self.x, self.originx * self.maxl)
    
    def act(self, a):
        """调整肌肉长度"""
        # 限制动作幅度，防止极端值
        a = max(min(a, self.stride * 10), -self.stride * 10)
        self.x += a
        self.regulation()
    
    def actdisp(self, a):
        """执行离散动作（0：收缩，1：舒张）"""
        # 获取当前实际物理距离
        current_dist = distant(self.p1, self.p2)
        
        # 安全检查：根据当前物理状态动态调整stride
        safe_stride = self.stride
        if a == 1 and current_dist > 0.95 * self.originx * self.maxl:
            # 如果已经接近最大长度，减小拉伸步长
            safe_stride = self.stride * 0.5
        elif a == 0 and current_dist < 1.05 * self.originx * self.minl:
            # 如果已经接近最小长度，减小收缩步长
            safe_stride = self.stride * 0.5
            
        # 执行动作，使用安全步长
        if a:
            self.x += safe_stride
        else:
            self.x -= safe_stride
            
        # 强制限制在安全范围
        self.regulation()
    
    def run(self):
        """计算肌肉产生的力并应用"""
        try:
            # 检查两物体距离是否合理
            current_dist = distant(self.p1, self.p2)
            if current_dist > 1e6:
                print("警告: 肌肉过度拉伸")
                return  # 不计算力，避免溢出
                
            self.p1.resilience(self.x, self.k, self.p2)
            
            # 计算阻尼力
            dv = [self.p1.v[0] - self.p2.v[0],
                self.p1.v[1] - self.p2.v[1],
                self.p1.v[2] - self.p2.v[2]]
            dp = [self.p1.p[0] - self.p2.p[0],
                self.p1.p[1] - self.p2.p[1],
                self.p1.p[2] - self.p2.p[2]]
            
            # 检查值是否合理
            if any(abs(v) > 1e6 for v in dv + dp):
                print("警告: 肌肉速度或位置差异过大")
                return
                
            # 计算速度在距离向量上的投影（安全版本）
            try:
                dp_magnitude = current_dist
                if dp_magnitude < 1e-10:  # 避免除以接近零的值
                    dk = 0
                else:
                    dk = sum([dv[i] * dp[i] for i in range(3)]) / dp_magnitude
                    # 限制dk的大小
                    dk = max(min(dk, 1e6), -1e6)
            except:
                dk = 0
                
            # 应用阻尼力
            self.p1.force2(dk * self.dampk, self.p2.p)
            self.p2.force2(dk * self.dampk, self.p1.p)
        except Exception as e:
            print(f"肌肉运行错误: {e}")


class Skeleton:
    """骨架类：保持两个物理点之间的固定距离"""
    
    def __init__(self, p1, p2, x = None, k = 1000, dampk = 20):
        """
        创建骨架
        
        参数:
            p1, p2: 连接的两个物理点
            x: 骨架长度
            k: 弹簧劲度系数
            dampk: 阻尼系数
        """
        self.p1 = p1
        self.p2 = p2
        
        # 安全计算初始长度
        try:
            self.x = distant(p1, p2) if x is None else x
        except:
            print("骨架初始化出错，使用默认长度100")
            self.x = 100.0
            
        self.k = k
        self.dampk = dampk
    
    def run(self):
        """计算骨架产生的力并应用"""
        try:
            # 检查两物体距离是否合理
            current_dist = distant(self.p1, self.p2)
            if current_dist > 1e6:
                print("警告: 骨架过度拉伸")
                return  # 不计算力，避免溢出
                
            self.p1.resilience(self.x, self.k, self.p2)
            
            # 计算阻尼力
            dv = [self.p1.v[0] - self.p2.v[0],
                self.p1.v[1] - self.p2.v[1],
                self.p1.v[2] - self.p2.v[2]]
            dp = [self.p1.p[0] - self.p2.p[0],
                self.p1.p[1] - self.p2.p[1],
                self.p1.p[2] - self.p2.p[2]]
            
            # 检查值是否合理
            if any(abs(v) > 1e6 for v in dv + dp):
                print("警告: 骨架速度或位置差异过大")
                return
                
            # 计算速度在距离向量上的投影（安全版本）
            try:
                dp_magnitude = current_dist
                if dp_magnitude < 1e-10:  # 避免除以接近零的值
                    dk = 0
                else:
                    dk = sum([dv[i] * dp[i] for i in range(3)]) / dp_magnitude
                    # 限制dk的大小
                    dk = max(min(dk, 1e6), -1e6)
            except:
                dk = 0
                
            # 应用阻尼力
            self.p1.force2(dk * self.dampk, self.p2.p)
            self.p2.force2(dk * self.dampk, self.p1.p)
        except Exception as e:
            print(f"骨架运行错误: {e}")


class Creature:
    """生物体类：包含物理点、肌肉和骨架组成的结构"""
    
    def __init__(self, phylist, musclelist, skeletonlist):
        """
        创建生物体
        
        参数:
            phylist: 物理点列表
            musclelist: 肌肉列表
            skeletonlist: 骨架列表
        """
        self.phys = phylist
        self.muscles = musclelist
        self.skeletons = skeletonlist
    
    def run(self):
        """运行生物体的物理模拟"""
        for i in self.muscles:
            i.run()
        for i in self.skeletons:
            i.run()
    
    def getstat(self, in3d = True, pk = 1, vk = 1, ak = 1, mk = 1, midform = True, conmid = False):
        """获取生物体状态向量，用于强化学习"""
        try:
            s = []
            d = 3 if in3d else 2
            mid = [0, 0, 0]
            
            if midform:
                # 安全计算中心点
                phy_count = len(self.phys)
                if phy_count == 0:
                    mid = [0, 0, 0]
                else:
                    for i in self.phys:
                        # 限制极端值
                        px = max(min(i.p[0], 1e6), -1e6)
                        py = max(min(i.p[1], 1e6), -1e6)
                        pz = max(min(i.p[2], 1e6), -1e6)
                        
                        mid[0] += px
                        mid[1] += py
                        mid[2] += pz
                    mid = [mid[j] / phy_count for j in range(3)]
            
            for i in self.phys:
                # 限制极端值
                px = max(min(i.p[0], 1e6), -1e6)
                py = max(min(i.p[1], 1e6), -1e6)
                pz = max(min(i.p[2], 1e6), -1e6)
                
                s += [(px - mid[j]) * pk for j in range(d)]
                
                # 限制速度值
                vx = max(min(i.v[0], 1e6), -1e6)
                vy = max(min(i.v[1], 1e6), -1e6)
                vz = max(min(i.v[2], 1e6), -1e6)
                
                s += [j * vk for j in [vx, vy, vz][:d]]
                
                # 限制加速度值
                ax = max(min(i.axianshi[0], 1e6), -1e6)
                ay = max(min(i.axianshi[1], 1e6), -1e6)
                az = max(min(i.axianshi[2], 1e6), -1e6)
                
                s += [ac * ak for ac in [ax, ay, az][:d]]
            
            if conmid:
                s += mid
            
            for i in self.muscles:
                # 限制肌肉长度值
                muscle_x = max(min(i.x, 1e6), -1e6)
                s.append(muscle_x * mk)
            
            return torch.tensor(s, dtype = torch.float32)
        except Exception as e:
            print(f"获取状态出错: {e}")
            # 返回合理默认值
            return torch.zeros(100, dtype=torch.float32)
    
    def act(self, a):
        """执行连续动作"""
        for i in range(min(len(self.muscles), len(a))):
            self.muscles[i].act(a[i])
    
    def actdisp(self, a):
        """执行离散动作"""
        for i in range(min(len(self.muscles), len(a))):
            self.muscles[i].actdisp(a[i])


class Environment:
    """环境类：包含生物体与物理世界的交互"""
    
    def __init__(self, creaturelist, in3d = False, g = 100, dampk = 0, groundhigh = 0, groundk = 1000,
                 grounddamp = 100, friction = 100, randsigma = 0.1):
        """
        创建环境
        
        参数:
            creaturelist: 生物体列表
            in3d: 是否为3D环境
            g: 重力加速度
            dampk: 空气阻尼系数
            groundhigh: 地面高度
            groundk: 地面刚度
            grounddamp: 地面阻尼
            friction: 摩擦系数
            randsigma: 随机噪声标准差
        """
        self.creatures = creaturelist
        self.g = g
        self.in3d = in3d
        self.dampk = dampk
        self.ground = groundhigh
        self.groundk = groundk
        self.grounddamp = grounddamp
        self.friction = friction
        self.sigma = randsigma
        
        # 初始随机扰动
        for i in self.creatures:
            for j in i.phys:
                j.v[0] += random.gauss(0, self.sigma)
                j.v[1] += random.gauss(0, self.sigma)
                if self.in3d:
                    j.v[2] += random.gauss(0, self.sigma)
    
    def run(self):
        """运行环境物理模拟"""
        try:
            for c in self.creatures:
                c.run()
                for p in c.phys:
                    # 检查位置是否有效
                    if any(abs(val) > 1e6 for val in p.p):
                        print("警告: 物体位置异常")
                        continue
                        
                    # 施加重力
                    p.force([0, -self.g, 0])
                    # 施加阻尼力
                    damp(p, self.dampk)
                    
                    # 地面碰撞检测
                    if p.p[1] - self.ground < 0:
                        p.color = "red"
                        p.r = 3
                        deep = (p.p[1] - self.ground)
                        # 确保浸入深度不会产生过大的力
                        deep = max(deep, -1000)  # 限制最大浸入深度
                        
                        # 地面弹力
                        p.force([0, -self.groundk * deep, 0])
                        # 地面阻尼
                        vy = max(min(p.v[1], 1e4), -1e4)  # 限制垂直速度
                        p.force([0, -self.grounddamp * vy, 0])
                        # 地面摩擦力
                        vx = max(min(p.v[0], 1e4), -1e4)  # 限制水平速度
                        vz = max(min(p.v[2], 1e4), -1e4)  # 限制水平速度
                        p.force([vx * deep * self.friction, 0, vz * deep * self.friction])
                    else:
                        p.color = "black"
                        p.r = 1
        except Exception as e:
            print(f"环境运行出错: {e}")
    
    def step(self, t):
        """环境前进一步"""
        try:
            self.run()
            Phy.run(t)
        except Exception as e:
            print(f"环境步进出错: {e}")
    
    def show(self, n = 1000, fps = 30):
        """显示环境模拟"""
        try:
            Phy.tready()
            
            for i in range(n):
                start_time = time.perf_counter()
                
                # 安全步进
                try:
                    self.step(0.001)
                except Exception as e:
                    print(f"步进错误: {e}")
                    break
                
                # 检查所有物体位置是否合理
                invalid_position = False
                for c in self.creatures:
                    for p in c.phys:
                        if any(abs(val) > 1e6 for val in p.p):
                            invalid_position = True
                            break
                
                if invalid_position:
                    print("物体位置异常，终止模拟")
                    break
                
                # 绘制地面
                turtle.goto(-800, self.ground)
                turtle.pendown()
                turtle.goto(800, self.ground)
                turtle.penup()
                
                # 安全绘制
                try:
                    Phy.tplay()
                except Exception as e:
                    print(f"绘制错误: {e}")
                
                # 控制帧率
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, 1 / fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            print(f"显示模拟出错: {e}")


def box1(scale = 1.0):
    """
    创建box1盒子模型
    
    参数:
        scale: 比例因子，控制盒子大小
    """
    # 创建4个物理点，形成矩形
    p = [Phy(1, [0, 0, 0], [-50 * scale, 0, 0]),
         Phy(1, [0, 0, 0], [-50 * scale, 100 * scale, 0]),
         Phy(1, [0, 0, 0], [50 * scale, 100 * scale, 0]),
         Phy(1, [0, 0, 0], [50 * scale, 0, 0])]
    
    # 创建顶部横向骨架
    sk = [Skeleton(p[1], p[2], k = 1200 * scale)]
    
    # 创建四个肌肉连接
    m = [Muscle(p[0], p[1], x = 100 * scale, k = 1000 * scale, maxl = 1.3, minl = 0.7, stride = 3 * scale),
         Muscle(p[0], p[2], x = math.sqrt(2 * 100 * 100) * scale, k = 800 * scale, maxl = 1.4, minl = 0.6,
                stride = 3 * scale),
         Muscle(p[3], p[1], x = math.sqrt(2 * 100 * 100) * scale, k = 800 * scale, maxl = 1.4, minl = 0.6,
                stride = 3 * scale),
         Muscle(p[3], p[2], x = 100 * scale, k = 1000 * scale, maxl = 1.3, minl = 0.7, stride = 3 * scale)]
    
    # 创建生物体
    c = Creature(p, m, sk)
    return c

# 测试代码
if __name__ == "__main__":
    # 创建盒子模型
    box = box1(scale=1.0)
    
    # 创建环境
    env = Environment([box], g=650, groundhigh=-50, groundk=10000,
                      grounddamp=100, friction=750, randsigma=0.05)
    
    # 显示模拟
    env.show(n=2000, fps=60)