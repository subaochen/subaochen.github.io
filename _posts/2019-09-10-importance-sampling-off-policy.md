---
title: 重要性采样和off-policy
type: post
categories:
- Reinforcement Learning
layout: post
date: 2019-09-10
tags: [deeplearning,reinforcement learning,importance sampling,off policy]
status: publish
published: true
use_math: true
comments: true
---

# 重要性采样求均值

从蒙特卡洛积分说起很容易理解重要性采样，参见这篇文章：[从蒙特卡洛积分到重要性采样](https://zhuanlan.zhihu.com/p/41217212)。[这里](https://www.jianshu.com/p/3d30070932a8)关于重要性采样的解释也很精彩，尤其是：

> 图中p(z)与f(z)的关系，p(z)是一种分布，是相对于z轴的采样点而言的，比如在红色的两个驼峰处，z的取点比较多，在其他地方z的取点就比较少，这叫样本分布服从p(z)。对于f(z)是一种映射关系，将z值映射到其他维度。比如我们熟悉的y = f(x)，将x映射到y。我们所说的求均值就是求f(z)的均值。

通过重要性采样，原本难以描述的$$p(x)$$就转化为容易计算的$$q(x)$$了：
$$
\begin{align}
\mathbb{E}[f]&=\int f(z)p(z)dz\\
&=\int f(z)\frac{p(z)}{q(z)}q(z)dz\\
&\approx\frac{1}{N}\sum_{k=1}^{N}\frac{p(z^{(k)})}{q(z^{(k)})}f(z^{(k)})
\end{align}
$$
其中，$$\frac{p(z)}{q(z)}$$即为重要性权重，求$$f(z)$$服从$$p(z)$$分布的均值的问题就转化为$$f(z)\frac{p(z)}{q(z)}$$服从$$q(z)$$分布的均值问题了。

# off-policy

