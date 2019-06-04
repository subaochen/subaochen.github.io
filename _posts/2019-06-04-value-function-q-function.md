---
title: MDP学习笔记-价值函数
type: post
categories:
- deeplearning
layout: post
date: 2019-06-03
tags: [deeplearning,mdp,reinforcement learning]
status: publish
published: true
use_math: true
comments: true

---
# 强化学习的目标

强化学习的目标是最大化Rewards。这里的Rewards不仅仅包括immediately reward，也包括后续的一系列reward。通常使用$$G_t$$来表示Rewards的累积结果：

$$
G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\ldots=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}
$$

其中，$$\gamma$$为衰减因子。

$$G_t$$可以表示为迭代的形式（递推公式）：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\ldots\\
=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\gamma^2R_{t+4}+\ldots)\\
=R_{t+1}+\gamma G_{t+1}
$$

# 策略函数

策略函数$$\pi(a\mid s)$$是指Agent在状态s下选取动作a的概率，因此$$\pi(a\mid s)$$是一个2元的概率函数，表示从状态s到动作a的映射概率。

$$\pi(a\mid s)$$是强化学习中需要优化的参数之一：在迭代优化过程中，需要不断调整$$\pi(a\mid s)$$使得最终的结果胜率最高。

# value function和Q function

$$v_{\pi}(s)$$表达了状态s在策略$$\pi$$下的好坏（价值），可以使用return的累积（期望）来衡量：

$$
v_{\pi}(s)=\mathbb{E_{\pi}}[G_t\mid S_t=s]=\mathbb{E_{\pi}}\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s\right]
$$

如下面的backup diagram所示：

![value-function-backup-diagram](/images/rl/value-function-backup-diagram.png)

假设当前处于状态s，即$$S_t=s$$。在策略$$\pi$$的作用下，从状态s可以有三个动作可以选择，每个动作进入Environment后会在2个后续状态s'及其对应的reward：r，于是我们评价状态s的价值，就需要将最后一层的所有reward进行加权平均（求期望），因为最后一层所有的reward都是状态s的可能结果。这是一个递归的过程。

因此，$$v_{\pi}(s)$$中的k表示计算第几层的return加权平均。（增加具体的例子说明？）

q function表达了状态s在策略$$\pi$$下，采取动作a的好坏（价值）：

$$
q_{\pi}(s,a)=\mathbb{E_{\pi}}[G_t\mid S_t=s,A_t=a]=\mathbb{E_{\pi}}\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s,A_t=a\right]
$$

下图有助于理解$$q_{\pi}(s,a)$$，见图中橙色线条标出部分。

![q-function-backup-diagram](/images/rl/q-function-backup-diagram.png)

从上图也可以看出$$q_{\pi}$$和$$v_{\pi}$$的关系：

$$
v_{\pi}(s)=\sum_{a}\pi(a\mid s)q_{\pi}(s,a)
$$

