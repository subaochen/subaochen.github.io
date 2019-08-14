---
title:  关于Exercise 3.8的疑问
type: post
categories:
- deeplearning
layout: post
date: 2019-08-14
tags: [deeplearning,reward,return,reinforcement learning]
status: publish
published: true
use_math: true
comments: true
---

《Reinforcement Learning, An Introduction》（2rd Edition）的Excercise 3.8是这样的：

> Suppose $$\gamma=0.5$$ and the following sequence of rewards is received $$R_1 = 1, R_2 = 2, R_3=6, R_4=3,$$ and $$R_5=2$$, with $$T=5$$. What are $$G_0,G_1,...,G_5$$? Hint: Work backwards.

在一份网上流传的Sutton本人给出的答案（未经验证）是这样的：

> $$G_0=2,G_1=3,G_2=2,G_3=\frac{1}{2},G_4=\frac{1}{8},G_5=0$$

毫无疑问，$$G_5=0$$。但是根据Return和Reward的递推公式：
$$
G_t=R_{t+1}+\gamma G_{t+1}
$$
很容易得到：
$$
G_4=R_5+\gamma G_5=2+0=2
$$
同理可得$$G_3=4,G_2=8,G_1=6,G_0=2$$，这和Sutton给出的答案差异太大了！

通过正向的Return计算公式：
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+\gamma^3R_{t+4}+...\gamma^{T-t-1}R_{T-t}
$$
可以得到同样的结论。

难道是哪里理解错了？还是Sutton给出了的答案不对？希望各位看到的同道指点一二，多谢！