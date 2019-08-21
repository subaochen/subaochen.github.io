---
title: DP学习笔记-策略增强
type: post
categories:
- Reinforcement Learning
layout: post
date: 2019-06-20
tags: [deeplearning,dynamic programming,reinforcement learning]
status: publish
published: true
use_math: true
comments: true
---

策略优化，或者寻找最优策略的三个步骤是：

1. 随机选择一个策略，比如uniform random policy，计算每个状态的值函数。这个随机选择的策略不一定（一般来说不是）最优策略，在这里计算状态的价值函数的目的是为了下一步policy improvement提供评判的**基准**。
2. 在第一步的基础上，在每一个状态都采取一个新的策略使得新策略下状态的价值函数都大于老策略下的状态价值函数。通常这个新的策略选择greedy policy。
3. 以上两个步骤循环迭代，直到策略稳定，即动作不再发生变化。

在<https://subaochen.github.io/deeplearning/2019/06/16/policy-evaluation/>中记录的是这三个步骤中的第一个：策略评估的学习笔记，在这里补充完整策略迭代的完整内容。

# policy improvement

policy improvement的关键是如何保证对策略的更新是有效的，即使得在新的策略下总的回报更高。这个问题可以转换为：如果能够确保（证明）新策略使得**每个状态**的价值都大于（不小于）老的策略，则新策略就是更好的策略；如果这个过程持续下去，则最终能够找到最优的策略。

考察greedy策略是否满足这个要求。greedy策略是指，在每个状态$$s$$的下一步要采取的动作由下式给出：

$$
\pi^{'}(s)=\underset{a}{\operatorname{argmax}} q_{\pi}(a,s)
$$

也就是说，在状态$$s$$下，找出所有可能的动作$$a$$，然后计算每个动作$$a$$的价值函数$$q_{\pi}(a,s)$$，选择使得$$a$$的动作价值函数最大的$$a$$作为优化的输出。注意，这里得到的a可能有多个，也未必是最终的最优方向。

简单的说，greedy策略是往前看一步，简单粗暴的选择动作价值最大的那个方向继续探索。

假设老的策略记为$$\pi$$，greedy策略记为$$\pi^{'}$$，则我们需要要证明：

$$
v_{\pi}(s)\leq v_{\pi^{'}}(s),\forall s\in\mathcal{S}
$$

考虑到greedy策略的目的是在下一步采取最大价值的动作， 我们可以先考察$$q_{\pi^{'}}(s,a)$$。由于存在如下的关系：

$$
v_{\pi}(s)=\sum\pi(a\mid s)q_{\pi}(s,a)
$$

当采用greedy策略时，权重$$\pi(a\mid s)$$会向greedy策略选中的动作集中。假设在greedy策略下状态s的动作空间的维度为m，$$m\ge 1$$，则有$$\frac{1}{m}\times q_{\pi}(s,\pi^{'}(s))=v_{\pi}(s)$$（如果m>1则意味着greedy策略找到了m个动作都使得$$q_{\pi^{'}}(s,a)$$取得最大值，显然此时这m个动作的$$q_{\pi}(s,a)$$是相等的），因此显然有：

$$
q_{\pi}(s,\pi^{'}(s))\ge v_{\pi}(s)\tag{0}
$$

下面逐步展开$$q_{\pi}(s,\pi^{'}(s))$$即可证明在greedy策略下，$$v_{\pi^{'}}\ge v_{\pi}(s)$$：

$$
\begin{align}
v_{\pi}(s)&\le q_{\pi}(s,\pi^{'}(s))\\
&=\mathbb{E_{\pi}}\left[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_t=s,A_t=\pi^{'}(s)\right]\tag{1-1}\\ 
&=\mathbb{E_{\pi^{'}}}\left[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_t=s\right]\tag{1-2}\\
&\le\mathbb{E_{\pi^{'}}}\left[R_{t+1}+\gamma q_{\pi}(S_{t+1},\pi^{'}(S_{t+1}))\mid S_t=s\right]\tag{1-3}\\
&=\mathbb{E_{\pi^{'}}}\left[R_{t+1}+\gamma\mathbb{E_{\pi^{'}}}\left[R_{t+2}+\gamma v_{\pi}(S_{t+2})\mid S_{t+1},A_{t+1}=\pi^{'}(S_{t+1})\right]\mid S_t=s\right]\tag{1-4}\\
&=\mathbb{E_{\pi^{'}}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2v_{\pi}(S_{t+2})\mid S_t=s\right]\tag{1-5}\\
&\le\mathbb{E_{\pi^{'}}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2v_{\pi}(S_{t+2})+\gamma^3 v_{\pi}(S_{t+3})\mid S_t=s\right]\tag{1-6}\\
\vdots\\
&\le\mathbb{E_{\pi^{'}}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\ldots\mid S_t=s\right]\tag{1-7}\\
&=v_{\pi^{'}}(s)
\end{align}
$$

上式中的每一步都值得仔细玩味：

* 式1-1表示在老的策略$$\pi$$的基础上更新为greedy策略，即在状态s中找到$$q_{\pi}(s,a)$$最大的前进方向$$\pi^{'}(s)$$。此时计算$$q_{\pi}(s,\pi^{'}(s))$$依然以策略$$\pi$$为基础，包括求期望，取得状态$$s'$$的价值函数$$v_{\pi}(s')$$即$$v_{\pi}(S_{t+1})$$等。也就是说，在式1-1中只是根据greedy策略选择了最佳的动作方向，其他和策略$$\pi$$没有差别，依然按照策略$$\pi$$来计算状态和动作的价值函数，这就是为什么$$\mathbb{E}$$的下标和$$v$$的下标为$$\pi$$的原因。
* <font color="red">式1-2没太看明白？</font>其大概的意思是，由于采用了greedy策略，参与期望计算的$$q_{\pi}(s,a)$$要少一些，因此可以记为在策略$$\pi^{'}$$下求期望：这是一种符号的代换？
* 将公式0带入式1-2可得式1-3。
* 将式1-1带入式1-3可得式1-4。
* 在式1-4中，期望的期望还是其本身，因此可以将里层的期望符号去掉，得到式1-5。但是，<font color="red">$v_{\pi}(S_{t+2})$的小尾巴（$S_{t+1},A_{t+1}=\pi^{'}(S_{t+1})$）是怎么甩掉的？</font>
* 式1-6和1-7是以上过程的循环。

**结论**：greedy策略可以实现policy improvement。有没有其他策略也可以实现policy improvement呢？作者没有阐述，有待进一步观察和研究。

# policy improvement的案例解析

还是以小小方格世界（grid world，参见：[DP学习笔记-策略评估](https://subaochen.github.io/deeplearning/2019/06/16/policy-evaluation/)）为例子，通过greedy policy来看一下是什么情况。（下面的计算过程实际上是value iteration，不是policy iteration。）

首先，初始化方格世界，采取uniform random policy，初始化所有状态的价值函数为0，如下图最左边列所示。此时，计算所有单元格的动作价值函数$$q(s,a)=\sum p(r,s'\mid s,a)(r+\gamma v(s'))$$：

$$
\begin{align}
q(1,left)&=-1+1\times0=-1\\
q(1,right)&=-1+1\times0=-1\\
q(1,up)&=-1+1\times0=-1\\
q(1,down)&=-1+1\times0=-1\\
q(2,left)&=-1+1\times0=-1\\
\ldots
\end{align}
$$

都是-1，因此在初始状态下，所有状态的动作都是均衡的4个方向，此时greedy policy没有任何效果，如下图中间列所示。

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/grid-world-pi-0.png)

第一轮迭代，首先计算状态价值函数，如下图左列所示。在此基础上，可以计算各个状态的动作价值函数如下：

$$
\begin{align}
q(1,left)&=-1+1\times0=-1\\
q(1,right)&=-1+1\times-1=-2\\
q(1,up)&=-1+1\times-1=-2\\
q(1,down)&=-1+1\times-1=-2\\
q(2,left)&=-1+1\times-1=-2\\
\ldots
\end{align}
$$

因此，根据greedy policy的原则，状态1的最佳动作应该是left，以此类推，greedy policy作用下的动作分布图如下所示。

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/grid-world-pi-1.png)

第二轮迭代，首先根据第二轮产生的动作分布图计算状态价值函数：

$$
\begin{align}
v(1)&=1\times(-1+1\times0)=-1\\
v(2)&=0.25\times(-1+1\times-1)+0.25\times(-1+1\times-1)+\ldots=-2\\
v(3)&=\ldots=-2\\
v(4)&=\ldots=-1\\
\ldots
\end{align}\\
$$

然后根据状态价值函数可以计算出每个$$q(s,a)$$（以状态2、5为例，状态1,4,11,14无需重复计算）：

$$
\begin{align}
q(2,left)&=-1+1\times-1=-2\\
q(2,right)&=-1+1\times-2=-3\\
q(2,up)&=-1+1\times-2=-3\\
q(2,down)&=-1+1\times-2=-3\\
q(5,left)&=-1+1\times-1=-2\\
q(5,right)&=-1+1\times-2=-3\\
q(5,up)&=-1+1\times-1=-2\\
q(5,down)&=-1+1\times-1=-3\\
\end{align}
$$

因此状态2在greedy policy下的最佳动作方向为left，状态5在greedy policy下的最佳动作为{left,up}，其他状态的最佳动作方向以此类推可以算出，如下图所示：

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/grid-world-pi-2.png)

第三轮迭代不再赘述，结果如下图所示（有趣的是，第三轮的状态价值函数没有发生变化。可以预见，此后的迭代也不会影响状态价值函数。）：

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/grid-world-pi-3.png)

实际上，经过3轮迭代已经找到了最佳策略（optimal policy），再进行更多轮次的迭代没有意义。

后记：在学习value iteration的时候发现，上面的计算方法是“半吊子”，既不是policy interation，也不是value iteration。policy iteration不使用贪心算法，而value interation是首先计算出收敛的状态价值函数，然后一次性根据贪心算法计算出最优策略。不过，针对这种“半吊子”算法，如果不是在每次迭代中求最优策略，而是间隔几次迭代求一次最优策略（根据贪心算法），是不是效率也蛮高的呢？



