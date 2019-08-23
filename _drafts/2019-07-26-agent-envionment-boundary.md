---
title: 学习笔记-智能体和环境的边界
type: post
categories:
- deeplearning
layout: post
date: 2019-07-26
tags: [deep learning,reinforcement learning, agent, environment]
status: publish
published: true
use_math: true
comments: true
---

# 智能体和环境的边界

sutton说：

> In reinforcement learning, as in other kinds of learning, such representational choices are at present more art than science.

这里的`representational choices`是说如何描述问题，或者学究一点，如何建模。sutton的观点多少有些“玄学”，把强化学习建模的过程比作“艺术创作”。

强化学习建模的第一步就是能够抽象出智能体和环境两个部分，进而找出<action, status, reward>的具体形式，然后才能借助MDP分析和解决问题。因此，找到智能体和环境的边界所在是成功的第一步，继续引用sutton的原话：

> The general rule we follow is that anything can not be changed arbitrarily by the agent is considered to be outside of it and thus part of environment.

也就是说，智能体和环境的边界在于：凡是智能体无法任意摆弄的部分都属于环境。这里的关键是“任意摆弄”，而不是智能体的视野、想法等。智能体必须是一个自己能够完全支配的对象，继续引用sutton的原话：

> The agent-environment boundary represents the limit of the agent's `absolute control`, not of its knowledge.

# 机械手臂的例子

假设我们要训练一个pick-and-place的机械手臂，我们的目标是尽量让机械手臂平滑的将物品移动到指定位置。在这个系统中的智能体和环境的划分可以为：

* 智能体：
* 环境：

这样可以识别出：

* action
* status
* reward

# 自动驾驶的例子

# 