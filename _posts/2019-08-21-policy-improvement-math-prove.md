---
title: policy improvement的数学证明
type: post
categories:
- Reinforcement Learning
layout: post
date: 2019-08-21
tags: [deeplearning,reinforcement learning,policy improvement]
status: publish
published: true
use_math: true
comments: true
---

对于Policy improvement作者没有给出严格的数学证明，其实至少从下面的角度可以严格证明Policy improvement的可行性。

首先证明期望的一个性质：

> 至少存在一个随机变量$$x$$，其值不小于随机变量$$X$$的期望$$\mathbb{E}[X]$$

证明如下：（参见《概率论基础教程》P251）

设$$f$$为定义在有限集$$A$$上的函数，假定我们对该函数的最大值感兴趣：

$$
m = \max_{s\in\mathcal{A}}f(s)
$$

为得到$$m$$的下界，令$$S$$为取值于$$A$$的随机元，显然有：$$m\ge f(S)$$，即：

$$
\mathbb{E}[m]\ge\mathbb{E}[f(S)]
$$

即：

$$
m\ge\mathbb{E}[f(S)]
$$

也就是说，函数$$f$$的最大值不小于$$f$$的期望，即在有限集$$A$$上，一定存在值（最大值）不小于$$f$$的期望。

有了以上的结论，我们考察$$v_{\pi}(s)$$和$$q_{\pi}(s,a)$$的关系：

$$
v_{\pi}(s)=\sum_{a}\pi(a\mid s)q_{\pi}(s,a)
$$

即在策略$$\pi$$下，**状态$$s$$的价值$$v_{\pi}(s)$$为$$q_{\pi}(s,a)$$的期望**，因此一定存在至少一个$$q_{\pi}(s,a)$$的最大值使得$$q_{\pi}(s,a)\ge v_{\pi}(s)$$。亦即，在动作$$A$$的集合中，一定存在至少一个动作$$a$$，其$$q(s,a)$$不小于状态$$s$$的价值$$v(s)$$。此结论对于任意$$s\in\mathcal{S}$$都成立。

我们定义对于任意$$s\in\mathcal{S}$$使得$$q_{\pi}(s,a)$$取最大值的策略为$$\pi^{'}$$，其动作为$$a'$$，即：

$$
a'=\pi^{'}(s)
$$

于是有：

$$
q_{\pi}(s,\pi^{'}(s))\ge v_{\pi}(s),\forall s\in\mathcal{S}\tag{1}
$$

利用这个结论，我们可以证明：

$$
v_{\pi^{'}}(s)\ge v_{\pi}(s),\forall s\in\mathcal{S}
$$

证明如下：

$$
\begin{align}
v_{\pi}(s)&\le q_{\pi}(s,\pi^{'}(s))\tag{2-1}\\
&=\mathbb{E}[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_t=s,A_t=\pi^{'}(s)]\tag{2-2}\\
&=\mathbb{E}_{\pi^{'}}[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_t=s]\tag{2-3}\\
&\le\mathbb{E}_{\pi^{'}}[R_{t+1}+\gamma q_{\pi}(S_{t+1},\pi^{'}(S_{t+1}))\mid S_t=s]\\
&=\mathbb{E}_{\pi^{'}}[R_{t+1}+\gamma\mathbb{E}_{\pi^{'}}[R_{t+2}+\gamma v_{\pi}(S_{t+2})\mid S_{t+1},A_{t+1}=\pi^{'}(S_{t+1})]\mid S_t=s]\\
&=\mathbb{E}_{\pi^{'}}[R_{t+1}+\gamma R_{t+2}+\gamma^2v_{\pi}(S_{t+2})\mid S_t=s]
\\
&=\mathbb{E}_{\pi^{'}}[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3v_{\pi}(S_{t+3})\mid S_t=s]\\
&\vdots\\
&\le\mathbb{E}_{\pi^{'}}[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\ldots\mid S_t=s]\\
&=v_{\pi^{'}}(s)
\end{align}
$$

在上面的证明中，值得注意的是第2步和第5步：

* 从第2步到第3步，这是一个等价变换，意为在策略$$\pi^{'}$$下的条件期望。进行这样的变换是为了方便后面推导出$$v_{\pi^{'}}(s)$$，因为状态价值函数是$$v_{\pi^{'}}(s)$$以状态$$S_t$$为条件对策略$$\pi^{'}$$的期望。
* 从第5步到第6步，是使用了期望的加法规则和“期望的期望是其本身”这个期望的性质。

也就是说，**对于任意的$$s\in\mathcal{S}$$，我们总是能够找到一个策略$$\pi^{'}$$使得其状态价值函数不小于策略$$\pi$$下的状态价值函数，即策略$$\pi^{'}\ge\pi$$**，这就是policy improvement的理论依据，具体的实现就是`greedy策略`：对于任意的$$s\in\mathcal{S}$$，找到使$$q(s,a)$$最大化的动作$$a$$即为最优策略。