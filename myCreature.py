# -*- coding: utf-8 -*-
"""
@File    : myCreature
@Author  : Limit
@Date    : 2025/4/24 15:27
@Description : 
"""

# Import statements (optional)
import torch
import random
import turtle
from physical import *


def distant(p1, p2):
    return ((p1.p[0] - p2.p[0]) ** 2 + (p1.p[1] - p2.p[1]) ** 2 + (p1.p[2] - p2.p[2]) ** 2) ** 0.5


def damp(p, k):
    p.force([-k * p.v[0], -k * p.v[1], -k * p.v[2]])


class Muscle:
    def __init__(self, p1, p2, x = None, k = 1000, maxl = 1.5, minl = 0.1, stride = 2, dampk = 20):
        self.p1 = p1
        self.p2 = p2
        self.x = distant(p1, p2) if x is None else x
        self.originx = self.x
        self.k = k
        self.dampk = dampk
        self.minl = minl
        self.maxl = maxl
        self.stride = stride
    
    def regulation(self):
        self.x = max(self.x, self.originx * self.minl)
        self.x = min(self.x, self.originx * self.maxl)
    
    def act(self, a):
        self.x += a
        self.regulation()
    
    def actdisp(self, a):
        if a:
            self.x += self.stride
        else:
            self.x -= self.stride
        self.regulation()
    
    def run(self):
        self.p1.resilience(self.x, self.k, self.p2)
        dv = [self.p1.v[0] - self.p2.v[0],
              self.p1.v[1] - self.p2.v[1],
              self.p1.v[2] - self.p2.v[2]]
        dp = [self.p1.p[0] - self.p2.p[0],
              self.p1.p[1] - self.p2.p[1],
              self.p1.p[2] - self.p2.p[2]]
        dk = sum([dv[0] * dp[0],
                  dv[1] * dp[1],
                  dv[2] * dp[2]]) / distant(self.p1, self.p2)
        self.p1.force2(dk * self.dampk, self.p2.p)
        self.p2.force2(dk * self.dampk, self.p1.p)


class Skeleton:
    def __init__(self, p1, p2, x = None, k = 1000, dampk = 20):
        self.p1 = p1
        self.p2 = p2
        self.x = distant(p1, p2) if x is None else x
        self.k = k
        self.dampk = dampk
    
    def run(self):
        self.p1.resilience(self.x, self.k, self.p2)
        dv = [self.p1.v[0] - self.p2.v[0],
              self.p1.v[1] - self.p2.v[1],
              self.p1.v[2] - self.p2.v[2]]
        dp = [self.p1.p[0] - self.p2.p[0],
              self.p1.p[1] - self.p2.p[1],
              self.p1.p[2] - self.p2.p[2]]
        dk = sum([dv[0] * dp[0],
                  dv[1] * dp[1],
                  dv[2] * dp[2]]) / distant(self.p1, self.p2)
        self.p1.force2(dk * self.dampk, self.p2.p)
        self.p2.force2(dk * self.dampk, self.p1.p)


class Creature:
    def __init__(self, phylist, musclelist, skeletonlist):
        self.phys = phylist
        self.muscles = musclelist
        self.skeletons = skeletonlist
    
    def run(self):
        for i in self.muscles:
            i.run()
        for i in self.skeletons:
            i.run()
    
    def getstat(self, in3d = True, pk = 1, vk = 1, ak = 1, mk = 1, midform = True, conmid = False):
        s = []
        d = 3 if in3d else 2
        mid = [0, 0, 0]
        if midform:
            for i in self.phys:
                mid[0] += i.p[0]
                mid[1] += i.p[1]
                mid[2] += i.p[2]
            mid[0] /= len(self.phys)
            mid[1] /= len(self.phys)
            mid[2] /= len(self.phys)
        for i in self.phys:
            s += [(i.p[j] - mid[j]) * pk for j in range(d)] + [j * vk for j in i.v[:d]] + [ac * ak for ac in
                                                                                           i.axianshi[:d]]
        if conmid:
            s += mid
        for i in self.muscles:
            s.append(i.x * mk)
        # 返回torch.Tensor代替Ten
        return torch.tensor(s, dtype = torch.float)
    
    def act(self, a):
        for i in range(len(self.muscles)):
            self.muscles[i].act(a[i])
    
    def actdisp(self, a):
        for i in range(len(self.muscles)):
            self.muscles[i].actdisp(a[i])


class Environment:
    def __init__(self, creaturelist, in3d = False, g = 100, dampk = 0, groundhigh = 0, groundk = 1000, grounddamp = 100,
                 friction = 100, randsigma = 0.1):
        self.creatures = creaturelist
        self.g = g
        self.in3d = in3d
        self.dampk = dampk
        self.ground = groundhigh
        self.groundk = groundk
        self.grounddamp = grounddamp
        self.friction = friction
        self.sigma = randsigma
        
        for i in self.creatures:
            for j in i.phys:
                j.v[0] += random.gauss(0, self.sigma)
                j.v[1] += random.gauss(0, self.sigma)
                if self.in3d:
                    j.v[2] += random.gauss(0, self.sigma)
    
    def run(self):
        for c in self.creatures:
            c.run()
            for p in c.phys:
                p.force([0, -self.g, 0])
                damp(p, self.dampk)
                
                if p.p[1] - self.ground < 0:
                    p.color = "red"
                    p.r = 3
                    deep = (p.p[1] - self.ground)
                    p.force([0, -self.groundk * deep, 0])
                    p.force([0, -self.grounddamp * p.v[1], 0])
                    p.force([p.v[0] * deep * self.friction, 0, p.v[2] * deep * self.friction])
                else:
                    p.color = "black"
                    p.r = 1
    
    def step(self, t):
        self.run()
        Phy.run(t)


def test():
    def f1():
        c.actdisp([1])
    
    def f0():
        c.actdisp([0])
    
    l = [Phy(1, [0, 0, 0], [-100, 100, 0]),
         Phy(1, [0, 0, 0], [100, 100, 0]),
         Phy(1, [0, 0, 0], [100, -100, 0]),
         Phy(1, [0, 0, 0], [-100, -100, 0])]
    sk = [Skeleton(l[0], l[1]),
          Skeleton(l[0], l[3]),
          Skeleton(l[2], l[3]),
          Skeleton(l[0], l[2], k = 100),
          Skeleton(l[1], l[3], k = 100)]
    m = [Muscle(l[1], l[2], stride = 3)]
    c = Creature(l, m, sk)
    Phy.tready()
    while True:
        c.run()
        turtle.onkeypress(f0, "[")
        turtle.onkeypress(f1, "]")
        turtle.listen()
        Phy.run(0.01)
        Phy.tplay()


def leg2():  # fail
    p = [Phy(1, [0, 0, 0], [0, 100, 0]),
         Phy(1, [0, 0, 0], [100, 100, 0]),  # 右上
         Phy(1, [0, 0, 0], [50, 50, 0]),
         Phy(1, [0, 0, 0], [100, 0, 0]),
         Phy(1, [0, 0, 0], [-100, 100, 0]),  # 左上
         Phy(1, [0, 0, 0], [-150, 50, 0]),
         Phy(1, [0, 0, 0], [-100, 0, 0])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[4]),
          Skeleton(p[1], p[4]),
          Skeleton(p[1], p[2]),
          Skeleton(p[2], p[3]),
          Skeleton(p[4], p[5]),
          Skeleton(p[5], p[6])]
    m = [Muscle(p[1], p[3]),
         Muscle(p[4], p[6]),
         Muscle(p[0], p[2]),
         Muscle(p[0], p[5])]
    c = Creature(p, m, sk)
    return c


def box():
    p = [Phy(1, [0, 0, 0], [-50, 0, 0]),
         Phy(1, [0, 0, 0], [-50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 0, 0]),
         Phy(1, [0, 0, 0], [50, 100, 0])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[1], p[2]),
          Skeleton(p[2], p[3])]
    m = [Muscle(p[0], p[2]),
         Muscle(p[1], p[3])]
    c = Creature(p, m, sk)
    return c


def box2():
    p = [Phy(1, [0, 0, 0], [-50, 0, 0]),
         Phy(1, [0, 0, 0], [-50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 0, 0])]
    sk = [Skeleton(p[1], p[2])]
    m = [Muscle(p[0], p[1]),
         Muscle(p[0], p[2]),
         Muscle(p[3], p[1]),
         Muscle(p[3], p[2])]
    c = Creature(p, m, sk)
    return c


def balance():
    p = [Phy(1, [0, 0, 0], [-50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 100, 0]),
         Phy(1, [0, 0, 0], [0, 0, 0]),
         Phy(1, [0, 0, 0], [0, 100, 0])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[3]),
          Skeleton(p[1], p[3])]
    m = [Muscle(p[0], p[2]),
         Muscle(p[1], p[2])]
    c = Creature(p, m, sk)
    return c


def balance2():
    p = [Phy(5, [0, 0, 0], [-50, 100, 0]),
         Phy(5, [0, 0, 0], [50, 100, 0]),
         Phy(1, [0, 0, 0], [0, 0, 0]),
         Phy(0.1, [0, 0, 0], [0, 100, 0])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[3], k = 10000),
          Skeleton(p[1], p[3], k = 10000)]
    m = [Muscle(p[0], p[2]),
         Muscle(p[1], p[2])]
    c = Creature(p, m, sk)
    return c


def balance3():
    p = [Phy(1, [0, 0, 0], [-50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 100, 0]),
         DingPhy(1, [0, 0, 0], [0, 0, 0]),
         Phy(0.1, [0, 0, 0], [0, 100, 0])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[3], k = 20000),
          Skeleton(p[1], p[3], k = 20000)]
    m = [Muscle(p[0], p[2]),
         Muscle(p[1], p[2])]
    c = Creature(p, m, sk)
    return c


def intrian():
    p = [Phy(1, [0, 0, 0], [-50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 100, 0]),
         Phy(1, [0, 0, 0], [0, 0, 0])]
    sk = []
    m = [Muscle(p[0], p[2]),
         Muscle(p[1], p[2]),
         Muscle(p[0], p[1])]
    c = Creature(p, m, sk)
    return c


def humanb():
    p = [Phy(1, [0, 0, 0], [25, 250, 0]),
         Phy(1, [0, 0, 0], [-25, 200, 0]),
         Phy(1, [0, 0, 0], [25, 150, 0]),
         Phy(1, [0, 0, 0], [-25, 100, 0]),
         Phy(1, [0, 0, 0], [25, 0, 0]),
         Phy(1, [0, 0, 0], [-25, 0, 0])]
    m = [Muscle(p[2], p[4]),
         Muscle(p[2], p[5]),
         Muscle(p[3], p[4]),
         Muscle(p[3], p[5])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[2]),
          Skeleton(p[1], p[2]),
          Skeleton(p[1], p[3]),
          Skeleton(p[2], p[3])]
    c = Creature(p, m, sk)
    return c


def insect():
    p = [Phy(1, [0, 0, 0], [-75, 100, 0]),
         Phy(1, [0, 0, 0], [-25, 100, 0]),
         Phy(1, [0, 0, 0], [25, 100, 0]),
         Phy(1, [0, 0, 0], [75, 100, 0]),
         Phy(1, [0, 0, 0], [-100, 50, 0]),
         Phy(1, [0, 0, 0], [-50, 50, 0]),
         Phy(1, [0, 0, 0], [0, 50, 0]),
         Phy(1, [0, 0, 0], [50, 50, 0]),
         Phy(1, [0, 0, 0], [100, 50, 0]),
         Phy(1, [0, 0, 0], [-75, 0, 0]),
         Phy(1, [0, 0, 0], [-25, 0, 0]),
         Phy(1, [0, 0, 0], [25, 0, 0]),
         Phy(1, [0, 0, 0], [75, 0, 0])]
    m = [Muscle(p[9], p[4]),
         Muscle(p[9], p[5]),
         Muscle(p[10], p[5]),
         Muscle(p[10], p[6]),
         Muscle(p[11], p[6]),
         Muscle(p[11], p[7]),
         Muscle(p[12], p[7]),
         Muscle(p[12], p[8])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[4]),
          Skeleton(p[0], p[5]),
          Skeleton(p[1], p[2]),
          Skeleton(p[1], p[5]),
          Skeleton(p[1], p[6]),
          Skeleton(p[2], p[3]),
          Skeleton(p[2], p[6]),
          Skeleton(p[2], p[7]),
          Skeleton(p[3], p[7]),
          Skeleton(p[3], p[8]),
          Skeleton(p[4], p[5]),
          Skeleton(p[5], p[6]),
          Skeleton(p[6], p[7]),
          Skeleton(p[7], p[8])]
    c = Creature(p, m, sk)
    return c


def box4():
    p = [Phy(1, [0, 0, 0], [-50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 100, 0]),
         Phy(1, [0, 0, 0], [50, 0, 0]),
         Phy(1, [0, 0, 0], [17, 0, 0]),
         Phy(1, [0, 0, 0], [-17, 0, 0]),
         Phy(1, [0, 0, 0], [-50, 0, 0])]
    m = [Muscle(p[0], p[2]),
         Muscle(p[0], p[3]),
         Muscle(p[0], p[4]),
         Muscle(p[0], p[5]),
         Muscle(p[1], p[2]),
         Muscle(p[1], p[3]),
         Muscle(p[1], p[4]),
         Muscle(p[1], p[5])]
    sk = [Skeleton(p[0], p[1])]
    c = Creature(p, m, sk)
    return c


def leg():
    p = [Phy(1, [0, 0, 0], [-50, 200, 0]),
         Phy(1, [0, 0, 0], [50, 200, 0]),
         Phy(1, [0, 0, 0], [-50, 140, 0]),
         Phy(1, [0, 0, 0], [50, 140, 0]),
         Phy(1, [0, 0, 0], [-50, 70, 0]),
         Phy(1, [0, 0, 0], [50, 70, 0]),
         Phy(1, [0, 0, 0], [-50, 0, 0]),
         Phy(1, [0, 0, 0], [50, 0, 0])]
    m = [Muscle(p[1], p[3]),
         Muscle(p[2], p[4]),
         Muscle(p[5], p[7])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[2]),
          Skeleton(p[1], p[2]),
          Skeleton(p[2], p[3]),
          Skeleton(p[3], p[4]),
          Skeleton(p[3], p[5]),
          Skeleton(p[4], p[5]),
          Skeleton(p[4], p[6]),
          Skeleton(p[5], p[6]),
          Skeleton(p[6], p[7])]
    c = Creature(p, m, sk)
    return c


def hat():
    p = [Phy(1, [0, 0, 0], [0, 150, 0]),
         Phy(1, [0, 0, 0], [-50, 30, 0]),
         Phy(1, [0, 0, 0], [50, 30, 0]),
         Phy(1, [0, 0, 0], [-50, 0, 0]),
         Phy(1, [0, 0, 0], [50, 0, 0])]
    m = [Muscle(p[1], p[3]),
         Muscle(p[1], p[4]),
         Muscle(p[2], p[3]),
         Muscle(p[2], p[4])]
    sk = [Skeleton(p[0], p[1]),
          Skeleton(p[0], p[2]),
          Skeleton(p[1], p[2])]
    c = Creature(p, m, sk)
    return c