# -*- coding: utf-8 -*-
"""
物理引擎模拟库 - 优化版本
"""

import turtle
import torch
import math
import numpy as np
from typing import List, Tuple, Optional, Union

class Phy:
    """物理点类：模拟物理运动、力学计算和可视化"""
    
    biao = []  # 记录所有创建的物理点
    rbiao = []  # 记录需要连成弹簧的点
    rbook = {}  # 储存弹簧长度
    zhenshu = 0  # 显示过的帧数
    
    def __init__(self, m: float, v: List[float], p: List[float], 
                 r: Optional[float] = None, color: str = "black", e: float = 0):
        """
        创建一个物理点
        
        参数:
            m: 质量大小
            v: [x,y,z]速度向量
            p: [x,y,z]位置向量
            r: 点的半径，默认为质量的0.3次方
            color: 点的颜色
            e: 电荷量
        """
        self.m = m
        self.v = list(v) if not isinstance(v, list) else v
        self.p = list(p) if not isinstance(p, list) else p
        self.a = [0, 0, 0]
        self.r = m ** 0.3 if r is None else r
        self.axianshi = self.a.copy()
        self.color = color
        self.e = e
        Phy.biao.append(self)
    
    def __repr__(self):
        return f"m={self.m}, v={self.v}, p={self.p}, a={self.axianshi}"
    
    def force(self, li: List[float]) -> None:
        """对点施加力向量"""
        for i in range(3):
            self.a[i] += li[i] / self.m
    
    def force2(self, lisize: float, p: List[float]) -> None:
        """对点施加指定大小的力，方向由目标位置决定"""
        mdx = [p[i] - self.p[i] for i in range(3)]
        odx = sum(x**2 for x in mdx) ** 0.5
        
        if odx < 1e-10:  # 防止除以零
            return
            
        li = [lisize * mdx[i] / odx for i in range(3)]
        self.force(li)
    
    def resilience(self, x: Optional[float] = None, k: float = 100, 
                  other = None, string: bool = False) -> None:
        """
        在两点间施加弹力
        
        参数:
            x: 弹簧原长，None时默认当前长度
            k: 劲度系数
            other: 弹簧另一端的点
            string: 弹力模型为绳型(True)或杆型(False)
        """
        # 计算两点距离
        key = (self, other)
        if x is None:
            if key not in Phy.rbook:
                dist = sum((other.p[i] - self.p[i])**2 for i in range(3)) ** 0.5
                Phy.rbook[key] = dist
            x = Phy.rbook[key]
        
        # 计算当前距离与原长差值
        curr_dist = sum((other.p[i] - self.p[i])**2 for i in range(3)) ** 0.5
        dx = curr_dist - x
        
        # 计算弹力大小
        if dx < 0 and string:
            lisize = 0  # 绳型模型，压缩时无力
        else:
            lisize = dx * k
            
        # 施加弹力
        self.force2(lisize, other.p)
        other.force2(lisize, self.p)
        
        # 记录弹簧连接，用于显示
        if (self, other) not in Phy.rbiao:
            Phy.rbiao.append((self, other))
    
    @classmethod
    def rread(cls, biao: List[dict]) -> None:
        """将弹力列表转为弹力"""
        for i in biao:
            i["self"].resilience(i["x"], i["k"], i["other"], i["string"])
    
    def bounce(self, k: float, other = "*") -> None:
        """碰撞处理：对指定点施以弹力"""
        if other == "*":
            other = Phy.biao
            
        for i in other:
            if i == self:
                continue
                
            dist = sum((i.p[j] - self.p[j])**2 for j in range(3)) ** 0.5
            if dist - self.r - i.r <= 0:
                self.resilience(self.r + i.r, k/2, i)
    
    @classmethod
    def gravity(cls, g: float) -> None:
        """对全部点施以万有引力"""
        for oout in cls.biao:
            for oin in cls.biao:
                if oout == oin:
                    continue
                    
                r = sum((oout.p[i] - oin.p[i])**2 for i in range(3)) ** 0.5
                if r < 1e-10:
                    continue
                    
                G = g * oout.m * oin.m / (r ** 2)
                oout.force2(G, oin.p)
    
    @classmethod
    def coulomb(cls, k: float) -> None:
        """对全部点施以静电力"""
        for oout in cls.biao:
            for oin in cls.biao:
                if oout == oin:
                    continue
                    
                r = sum((oout.p[i] - oin.p[i])**2 for i in range(3)) ** 0.5
                if r < 1e-10:
                    continue
                    
                f = -k * oout.e * oin.e / (r ** 2)
                oout.force2(f, oin.p)
    
    def electrostatic(self, k: float) -> None:
        """对点施以静电力"""
        for i in Phy.biao:
            if i == self:
                continue
                
            r = sum((self.p[j] - i.p[j])**2 for j in range(3)) ** 0.5
            if r < 1e-10:
                r = 1e-9
                
            f = -k * self.e * i.e / (r ** 2)
            self.force2(f, i.p)
    
    @classmethod
    def momentum(cls) -> List[float]:
        """计算全局动量和"""
        return [sum(i.v[j] * i.m for i in cls.biao) for j in range(3)]
    
    @classmethod
    def run(cls, t: float) -> None:
        """运行物理模型一步"""
        for dian in cls.biao:
            # 更新速度
            for i in range(3):
                dian.v[i] += dian.a[i] * t
            
            # 更新位置
            for i in range(3):
                dian.p[i] += dian.v[i] * t
            
            # 保存当前加速度，重置加速度
            dian.axianshi = dian.a.copy()
            dian.a = [0, 0, 0]
    
    @classmethod
    def hprun(cls, t: float) -> None:
        """高精度运行模型（半隐式欧拉法）"""
        for dian in cls.biao:
            # 更新位置（考虑加速度）
            for i in range(3):
                dian.p[i] += dian.v[i] * t + 0.5 * dian.a[i] * t ** 2
            
            # 更新速度
            for i in range(3):
                dian.v[i] += dian.a[i] * t
            
            # 保存当前加速度，重置加速度
            dian.axianshi = dian.a.copy()
            dian.a = [0, 0, 0]
    
    @classmethod
    def tready(cls) -> None:
        """初始化turtle显示"""
        turtle.tracer(0)
        turtle.penup()
        turtle.hideturtle()
    
    @classmethod
    def saveone(cls) -> tuple:
        """保存当前状态"""
        m, v, p, r, color, axianshi = [], [], [], [], [], []
        
        for i in cls.biao:
            m.append(i.m)
            v.append(i.v)
            p.append(i.p)
            r.append(i.r)
            color.append(i.color)
            axianshi.append(i.axianshi)
            
        rbiao = [(cls.biao.index(j[0]), cls.biao.index(j[1])) for j in cls.rbiao]
        
        return (tuple(m), tuple(v), tuple(p), tuple(r), 
                tuple(color), tuple(axianshi), tuple(rbiao))
    
    @classmethod
    def readone(cls, z: tuple) -> None:
        """读取保存的状态"""
        cls.biao = []
        cls.rbiao = []
        
        for j in range(len(z[0])):
            Phy(z[0][j], z[1][j], z[2][j], z[3][j], z[4][j])
            
        for i2 in range(len(cls.biao)):
            cls.biao[i2].axianshi = z[5][i2]
            
        for k in z[6]:
            cls.rbiao.append((cls.biao[k[0]], cls.biao[k[1]]))
    
    @staticmethod
    def xianxing(d: List[float], x: List[List[float]]) -> List[float]:
        """线性变换"""
        d_tensor = torch.tensor(d, dtype=torch.float)
        x_tensor = torch.tensor(x, dtype=torch.float)
        return torch.matmul(x_tensor, d_tensor).tolist()
    
    @staticmethod
    def reference(d1: List[float], dr: List[float]) -> List[float]:
        """参考系变化"""
        return [d1[i] - dr[i] for i in range(3)]
    
    @staticmethod
    def perspective(d: List[float], cam: List[float], k: float) -> List[float]:
        """透视变换"""
        d2 = Phy.reference(d, cam)
        d2[2] = 0.00001 if d2[2] == 0 else d2[2]
        return [d2[0] * k / d2[2], d2[1] * k / d2[2]]
    
    @staticmethod
    def shijiaox(fm: List[float], to: List[float]) -> List[List[float]]:
        """视角矢量x，在x-z平面上旋转坐标轴"""
        zl = ((to[0] - fm[0]) ** 2 + (to[2] - fm[2]) ** 2) ** 0.5
        if zl < 1e-10:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
        zx = -(to[0] - fm[0]) / zl
        zz = (to[2] - fm[2]) / zl
        rz = (zx ** 2 + zz ** 2) ** 0.5
        
        if rz < 1e-10:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
        xx = zz / rz
        xz = -zx / rz
        
        return [[xx, 0, xz], [0, 1, 0], [zx, 0, zz]]
    
    @staticmethod
    def shijiaoy(fm: List[float], to: List[float]) -> List[List[float]]:
        """视角矢量y，在y-z平面上旋转坐标轴"""
        zl = ((to[1] - fm[1]) ** 2 + (to[2] - fm[2]) ** 2) ** 0.5
        if zl < 1e-10:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
        zy = -(to[1] - fm[1]) / zl
        zz = (to[2] - fm[2]) / zl
        rz = (zy ** 2 + zz ** 2) ** 0.5
        
        if rz < 1e-10:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
        yy = zz / rz
        yz = -zy / rz
        
        return [[1, 0, 0], [0, yy, yz], [0, zy, zz]]
    
    @classmethod
    def shijiaoshi(cls, fm: List[float], to: List[float]) -> List[List[float]]:
        """视角矢量，旋转坐标轴至出发点正对着看向点"""
        mx = cls.shijiaox(fm, to)
        fm_new = cls.xianxing(fm, mx)
        to_new = cls.xianxing(to, mx)
        my = cls.shijiaoy(fm_new, to_new)
        
        mx_tensor = torch.tensor(mx, dtype=torch.float)
        my_tensor = torch.tensor(my, dtype=torch.float)
        return torch.matmul(my_tensor, mx_tensor).tolist()
    
    @classmethod
    def dotpos(cls, pos: List[float], c=None, x=None) -> List[float]:
        """计算坐标点经过变换后的位置"""
        if c is None:
            c = [0, 0, 0]
        if x is None:
            x = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
        return cls.xianxing(cls.reference(pos, c), x)
    
    @classmethod
    def tplay(cls, fps=1, a=False, v=False, c=None, x=None, 
              azoom=1, vzoom=1, k=None) -> None:
        """使用turtle显示物理模型"""
        # 透视变换函数
        toushi = lambda pos: cls.perspective(pos, [0, 0, 0], k) if k is not None else pos
        
        # 默认参考系和变换矩阵
        if c is None:
            c = DingPhy(0, [0, 0, 0], [0, 0, 0], 0)
        if x is None:
            x = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
        # 按帧率控制显示
        if cls.zhenshu % fps == 0:
            # 绘制弹簧连接
            for i in cls.rbiao:
                turtle.color("black")
                dr0 = cls.dotpos(i[0].p, c.p, x)
                dr1 = cls.dotpos(i[1].p, c.p, x)
                
                # 透视检查
                if k is not None and (dr0[2] <= 0 or dr1[2] <= 0):
                    continue
                    
                dr0 = toushi(dr0)
                dr1 = toushi(dr1)
                
                turtle.goto(dr0[0], dr0[1])
                turtle.pendown()
                turtle.goto(dr1[0], dr1[1])
                turtle.penup()
                
            cls.rbiao = []
            
            # 绘制物理点
            for i in cls.biao:
                d = cls.dotpos(i.p, c.p, x)
                
                # 透视检查
                if k is not None and d[2] <= 0:
                    continue
                    
                d2 = toushi(d)
                turtle.goto(d2[0], d2[1])
                
                # 点大小根据透视变化
                dot_size = i.r * 2 / d[2] * k if k is not None else i.r * 2
                turtle.dot(dot_size, i.color)
                
                # 绘制加速度向量
                if a:
                    da_pos = [
                        i.p[j] - c.p[j] + (i.axianshi[j] - c.axianshi[j]) * azoom 
                        for j in range(3)
                    ]
                    da = cls.xianxing(da_pos, x)
                    
                    if k is not None and da[2] <= 0:
                        continue
                        
                    da = toushi(da)
                    turtle.pencolor("red")
                    turtle.goto(d2[0], d2[1])
                    turtle.pendown()
                    turtle.goto(da[0], da[1])
                    turtle.penup()
                    turtle.pencolor("black")
                
                # 绘制速度向量
                if v:
                    dv_pos = [
                        i.p[j] - c.p[j] + (i.v[j] - c.v[j]) * vzoom 
                        for j in range(3)
                    ]
                    dv = cls.xianxing(dv_pos, x)
                    
                    if k is not None and dv[2] <= 0:
                        continue
                        
                    dv = toushi(dv)
                    turtle.pencolor("blue")
                    turtle.goto(d2[0], d2[1])
                    turtle.pendown()
                    turtle.goto(dv[0], dv[1])
                    turtle.penup()
                    turtle.pencolor("black")
            
            # 更新显示
            turtle.update()
            turtle.clear()
            
        cls.zhenshu += 1

    class camera:
        """相机类：用于3D场景观察"""
        
        def __init__(self, campos=None, lookpos=None, fix=True, k=300):
            """
            创建相机
            
            参数:
                campos: 相机位置
                lookpos: 注视点位置
                fix: 相机是否固定
                k: 视野系数
            """
            if campos is None:
                campos = [0, 0, -300]
            if lookpos is None:
                lookpos = [0, 0, 0]
                
            # 创建相机物理点
            if fix:
                self.cam = DingPhy(1, [0, 0, 0], campos)
            else:
                self.cam = Phy(1, [0, 0, 0], campos)
                
            # 计算相对注视方向
            l = sum((lookpos[i] - campos[i])**2 for i in range(3)) ** 0.5
            self.relalookpos = [(lookpos[i] - campos[i]) / l for i in range(3)]
            self.k = k
            
        def setlookpos(self, lookpos):
            """设置相机注视点"""
            l = sum((lookpos[i] - self.cam.p[i])**2 for i in range(3)) ** 0.5
            self.relalookpos = [(lookpos[i] - self.cam.p[i]) / l for i in range(3)]
            
        def dotposspace(self, pos):
            """计算点在相机空间中的位置"""
            lookto = [self.relalookpos[i] + self.cam.p[i] for i in range(3)]
            x = Phy.shijiaoshi(self.cam.p, lookto)
            return Phy.dotpos(pos, self.cam.p, x)
            
        def cdotpos(self, pos):
            """计算点在屏幕上的投影位置"""
            lookto = [self.relalookpos[i] + self.cam.p[i] for i in range(3)]
            x = Phy.shijiaoshi(self.cam.p, lookto)
            d = Phy.dotpos(pos, self.cam.p, x)
            
            if d[2] > 0:
                return Phy.perspective(d, [0, 0, 0], self.k)
            return None
            
        @classmethod
        def tready(cls):
            """初始化turtle显示"""
            Phy.tready()
            
        def tplay(self, a=False, v=False, azoom=1, vzoom=1, zuobiaoxian=False):
            """显示当前视图"""
            lookto = [self.relalookpos[i] + self.cam.p[i] for i in range(3)]
            x = Phy.shijiaoshi(self.cam.p, lookto)
            
            # 绘制坐标轴
            if zuobiaoxian:
                xian = [
                    Phy.xianxing([100, 0, 0], x),
                    Phy.xianxing([0, 100, 0], x),
                    Phy.xianxing([0, 0, 100], x)
                ]
                turtle.goto(xian[2][0], xian[2][1])
                turtle.dot(3, "red")
                
                for i in range(len(xian)):
                    turtle.pencolor("black")
                    turtle.goto(0, 0)
                    turtle.pendown()
                    turtle.goto(xian[i][0], xian[i][1])
                    turtle.penup()
                    
            # 显示场景
            Phy.tplay(a=a, v=v, azoom=azoom, vzoom=vzoom, 
                     c=self.cam, x=x, k=self.k)
                     
        def movecam(self, stepsize=1, camstepsize=0.02):
            """键盘控制相机移动"""
            # 前进
            def fw():
                dl = (self.relalookpos[0]**2 + self.relalookpos[2]**2) ** 0.5
                self.cam.p[0] += self.relalookpos[0] / dl * stepsize
                self.cam.p[2] += self.relalookpos[2] / dl * stepsize
            
            # 后退
            def bw():
                dl = (self.relalookpos[0]**2 + self.relalookpos[2]**2) ** 0.5
                self.cam.p[0] -= self.relalookpos[0] / dl * stepsize
                self.cam.p[2] -= self.relalookpos[2] / dl * stepsize
            
            # 向左移动
            def le():
                dl = (self.relalookpos[0]**2 + self.relalookpos[2]**2) ** 0.5
                self.cam.p[0] -= self.relalookpos[2] / dl * stepsize
                self.cam.p[2] += self.relalookpos[0] / dl * stepsize
            
            # 向右移动
            def ri():
                dl = (self.relalookpos[0]**2 + self.relalookpos[2]**2) ** 0.5
                self.cam.p[0] += self.relalookpos[2] / dl * stepsize
                self.cam.p[2] -= self.relalookpos[0] / dl * stepsize
            
            # 向上移动
            def zp():
                self.cam.p[1] += stepsize
            
            # 向下移动
            def zn():
                self.cam.p[1] -= stepsize
            
            # 视角向上
            def cu():
                self.relalookpos[1] += camstepsize
            
            # 视角向下
            def cd():
                self.relalookpos[1] -= camstepsize
            
            # 视角向左
            def cl():
                dl = (self.relalookpos[0]**2 + self.relalookpos[2]**2) ** 0.5
                self.relalookpos[0] -= self.relalookpos[2] / dl * camstepsize
                self.relalookpos[2] += self.relalookpos[0] / dl * camstepsize
            
            # 视角向右
            def cr():
                dl = (self.relalookpos[0]**2 + self.relalookpos[2]**2) ** 0.5
                self.relalookpos[0] += self.relalookpos[2] / dl * camstepsize
                self.relalookpos[2] -= self.relalookpos[0] / dl * camstepsize
            
            # 视角前进
            def zp2():
                self.relalookpos[2] += camstepsize
            
            # 视角后退
            def zn2():
                self.relalookpos[2] -= camstepsize
            
            # 放大视图
            def zin():
                self.k *= 1.1
            
            # 缩小视图
            def zout():
                self.k *= 0.9
            
            # 按键绑定
            turtle.onkeypress(fw, key="w")
            turtle.onkeypress(bw, key="s")
            turtle.onkeypress(le, key="a")
            turtle.onkeypress(ri, key="d")
            turtle.onkeypress(zp, key="space")
            turtle.onkeypress(zn, key="Control_L")
            turtle.onkeypress(cu, key="Up")
            turtle.onkeypress(cd, key="Down")
            turtle.onkeypress(cl, key="Left")
            turtle.onkeypress(cr, key="Right")
            turtle.onkeypress(zp2, key="u")
            turtle.onkeypress(zn2, key="o")
            turtle.onkeypress(zin, key="]")
            turtle.onkeypress(zout, key="[")
            turtle.listen()
    
    class tgraph:
        """图表显示类"""
        
        def __init__(self):
            """初始化图表"""
            self.biao = []
            self.zhenshu = 0
        
        def clean(self):
            """清空图表"""
            self.__init__()
        
        def draw(self, inx, iny, dis, chang=200, kx=1, ky=1, 
                tiao=1, color="black", phyon=True, bi=False):
            """
            绘制图表
            
            参数:
                inx: x坐标 (None表示自动递增)
                iny: y坐标
                dis: 坐标原点位置
                chang: 图表长度
                kx, ky: 放大系数
                tiao: 采样间隔
                color: 颜色
                phyon: 是否使用Phy.tplay
                bi: 是否画线
            """
            if not phyon:
                Phy.tready()
            
            # 按采样间隔记录数据
            if self.zhenshu % tiao == 0:
                self.biao.append([len(self.biao) if inx is None else inx, iny])
                
            # 限制数据点数量
            while len(self.biao) > chang:
                self.biao.pop(0)
            
            # 绘制图表
            if inx is None:
                if bi:
                    turtle.pencolor(color)
                    turtle.goto(dis[0], dis[1] + self.biao[0][1] * ky)
                    turtle.pendown()
                    
                for i in range(len(self.biao)):
                    turtle.goto(dis[0] + i * kx, dis[1] + self.biao[i][1] * ky)
                    turtle.dot(2, color)
                    
                if bi:
                    turtle.penup()
            else:
                if bi:
                    turtle.pencolor(color)
                    turtle.goto(dis[0] + self.biao[0][0] * kx, dis[1] + self.biao[0][1] * ky)
                    turtle.pendown()
                    
                for i in range(len(self.biao)):
                    turtle.goto(dis[0] + self.biao[i][0] * kx, dis[1] + self.biao[i][1] * ky)
                    turtle.dot(2, color)
                    
                if bi:
                    turtle.penup()
            
            # 更新屏幕
            if not phyon:
                turtle.update()
                turtle.clear()
                
            self.zhenshu += 1


class DingPhy(Phy):
    """固定物理点，不参与力的计算"""
    
    def __init__(self, m, v, p, r=None, color="black"):
        """创建固定物理点"""
        self.m = m
        self.v = list(v) if not isinstance(v, list) else v
        self.p = list(p) if not isinstance(p, list) else p
        self.a = [0, 0, 0]
        self.r = m ** 0.3 if r is None else r
        self.axianshi = [0, 0, 0]
        self.color = color


class Changjing:
    """场景类，管理和渲染物理对象"""
    
    allbiao = []  # 场景中的所有对象
    camara = [0, 0, -1]  # 相机位置
    k = 1  # 镜头放大参数
    
    @classmethod
    def tready(cls):
        """初始化turtle显示"""
        turtle.tracer(0)
        turtle.penup()
        turtle.hideturtle()
    
    @classmethod
    def view(cls, p, camara, k):
        """计算透视投影"""
        viewlength = max(camara[2] - p[2], 0.0000001)
        dx = (camara[0] - p[0]) / viewlength * k
        dy = (camara[1] - p[1]) / viewlength * k
        return (dx, dy)
    
    @classmethod
    def biaoupdate(cls):
        """根据Z轴排序渲染顺序"""
        cls.allbiao.sort(key=lambda x: x.p[2], reverse=True)
    
    @classmethod
    def play(cls, t):
        """显示并模拟一步"""
        for i in cls.allbiao:
            if i.p[2] <= cls.camara[2]:
                continue
            i.draw()
            
        turtle.update()
        turtle.clear()
        Phy.run(t)
    
    @classmethod
    def keymove(cls):
        """键盘控制相机移动"""
        def zf():
            cls.k *= 1.1
        
        def zb():
            cls.k *= 0.9
        
        def f():
            cls.camara[2] += 1
        
        def b():
            cls.camara[2] -= 1
        
        def l():
            cls.camara[0] -= 100
        
        def r():
            cls.camara[0] += 100
        
        def u():
            cls.camara[1] += 100
        
        def d():
            cls.camara[1] -= 100
        
        def reset(x, y):
            cls.k = 1
            cls.camara = [0, 0, -1]
        
        turtle.onkeypress(zf, key="=")
        turtle.onkeypress(zb, key="-")
        turtle.onkeypress(f, key="w")
        turtle.onkeypress(b, key="s")
        turtle.onkeypress(l, key="Left")
        turtle.onkeypress(r, key="Right")
        turtle.onkeypress(u, key="Up")
        turtle.onkeypress(d, key="Down")
        turtle.onscreenclick(reset)
        turtle.listen()


class object:
    """物理对象类，用于封装Phy点集合"""
    
    def __init__(self, color=(0, 0, 0)):
        """初始化物理对象"""
        self.biao = []
        self.color = color
        Changjing.allbiao.append(self)
    
    def tri(self, d, h, p, v=None, m=1, color="black"):
        """创建三角形对象"""
        if v is None:
            v = [0, 0, 0]
            
        self.biao = [
            Phy(m, v, [p[0], p[1], p[2]]),
            Phy(m, v, [p[0] + d, p[1], p[2]]),
            Phy(m, v, [p[0] + d/2, p[1] + h, p[2]]),
            Phy(m, v, p)
        ]
        
        self.color = color
        self.p = p
    
    def fang(self, r, p, v=None, m=1, color="black"):
        """创建正方形对象"""
        if v is None:
            v = [0, 0, 0]
            
        self.biao = [
            Phy(m, v, [p[0], p[1], p[2]]),
            Phy(m, v, [p[0] + r, p[1], p[2]]),
            Phy(m, v, [p[0] + r, p[1] + r, p[2]]),
            Phy(m, v, [p[0], p[1] + r, p[2]]),
            Phy(m, v, p)
        ]
        
        self.color = color
        self.p = p
    
    def cfang(self, c, f, p, v=None, m=1, color="black"):
        """创建长方形对象"""
        if v is None:
            v = [0, 0, 0]
            
        self.biao = [
            Phy(m, v, [p[0], p[1], p[2]]),
            Phy(m, v, [p[0] + c, p[1], p[2]]),
            Phy(m, v, [p[0] + c, p[1] + f, p[2]]),
            Phy(m, v, [p[0], p[1] + f, p[2]]),
            Phy(m, v, p)
        ]
        
        self.color = color
        self.p = p
    
    def draw(self):
        """绘制对象"""
        turtle.fillcolor(self.color)
        turtle.begin_fill()
        
        for i in self.biao:
            turtle.goto(Changjing.view(i.p, Changjing.camara, Changjing.k))
            
        turtle.end_fill()