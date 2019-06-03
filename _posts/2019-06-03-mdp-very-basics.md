---
title: MDP学习笔记-基本的交互过程
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

# 强化学习的基本环境和流程

强化学习中的MDP是Agent和Environment交互的过程，如下图所示：

![agent-envrionment](images/rl/agent-envrionment.png)

假设Agent的初始状态为$$S_0$$，Agent根据自身的策略（随机或者某种特定的策略）采取了动作$$A_0$$，Envrionment根据此动作给出了Agent的下一状态$$S_1$$和相应的Reward：$$R_1$$，Agent再根据$$S_1$$决定采取动作$$A_1$$，于是形成了如下的序列：

$$
S_0,A_0,S_1,R_1,A_1,S_2,R_2,A_2,...
$$

如果假设$$R_0 = 0$$的话，则序列为：

$$
S_0,R_0,A_0,S_1,R_1,A_1,S_2,R_2,A_2,...
$$

也就是说，<S,R,A>三元组构成了MDP的主要过程。

# dynamics funtion
如果我们把Agent和Environment分别看做黑盒，那么表达强化学习的交互过程至少应该包含下面的两个概率：

* 表示Agent的概率，$$p(a|s',r)$$，即Agent接收到状态s'和r后输出动作a的概率；
* 表示Environment的概率，$$p(s',r|a)$$，即Environment接收到动作a后输出状态s'和r的概率；

其中，a,s',r分别是随机变量$$A_t,S_t,R_t$$在时刻t的特定取值。在给定状态空间S和动作空间A的情况下，如果我们能够计算出所有的Agent概率和Environment的概率，则整个MDP的解空间是可以计算出来的。

也许为了统一化处理，作者定义了**dynamics function**来计算MDP的解空间（或许这是随机过程的一个概念？）：

$$
p(s',r|a,s)=P_r(S_t=s',R_t=r | A_{t-1}=a, S_{t-1}=s)
$$

dynamics function是对Environment的建模，即将Agent看做一个白盒（合理的假设，因为Agent对我们而言是完全可控的），则Agent的输入s和输出a对于Environment而言可以看做是Environment的输入，因此dynamics function的意义为：给定状态s和动作a，Environment输出下一时刻的状态s'及其相应的reward r的概率。








