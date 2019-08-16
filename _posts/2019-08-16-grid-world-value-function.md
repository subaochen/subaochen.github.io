---
title: MDP学习笔记-grid world问题的解析
type: post
categories:
- deeplearning
layout: post
date: 2019-08-16
tags: [deeplearning,mdp,value function,reinforcement learning,grid world]
status: publish
published: true
use_math: true
comments: true
---

# Example 3.5-grid world

下图是一个grid world，每个格子代表了一个状态，从每个格子出发可以有4个动作：up/left/down/right，每个动作机会均等，如果动作的结果超出了边界，则格子的状态保持不变，并且获得的奖励为-1；从A格子出发的任何动作都会导致状态转移到A'格子，并且获得奖励+10；从B格子出发的任何动作都会导致状态转移到B'格子，并且获得奖励+5；其他任何情况的奖励为0。假设$$\gamma=0.9$$，试计算每个格子的状态价值函数。

![grid world](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/images/rl/mdp/grid-world-5x5.png )

# 解析

这是一个典型的MDP问题，有限的状态空间，有限的动作空间。但是，我们面临两个棘手的问题：

* 根据Bellman等式，当前状态的价值依赖于下一个状态的价值，这是一个递归的过程。从何处开始，在何处结束？对于每一个状态都有不同的episode。
* 状态空间和动作空间如何表达？状态空间的表达容易想到使用格子的坐标来表示，比如格子A的状态可表示为坐标(0,1)，那么动作空间如何表达比较合适呢？

## Bellman等式的数值逼近求解

## 动作空间的表示方式

## 程序实现





完整源码请参见：[grid world源码](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/resources/grid_world.py)