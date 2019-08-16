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

下图是一个grid world，每个格子代表了一个状态，从每个格子出发可以有4个动作：up/left/down/right，每个动作机会均等，如果动作的结果超出了边界，则格子的状态保持不变，并且获得的奖励为-1；从A格子出发的任何动作都会导致状态转移到A'格子，并且获得奖励+10；从B格子出发的任何动作都会导致状态转移到B'格子，并且获得奖励+5；其他任何情况的奖励为0。试计算每个格子的状态价值函数。

![grid world](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/images/rl/mdp/grid-world-5x5.png )

# 解析



完整源码请参见：[grid world源码](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/resources/grid_world.py)