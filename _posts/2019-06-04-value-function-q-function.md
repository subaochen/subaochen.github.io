---
title: MDP学习笔记-价值函数
type: post
categories:
- deeplearning
layout: post
date: 2019-06-04
tags: [deeplearning,mdp,reinforcement learning]
status: publish
published: true
use_math: true
comments: true

---
# 强化学习的目标

强化学习的目标是最大化Rewards。这里的Rewards不仅仅包括immediately reward，也包括后续的一系列reward。通常使用Return（回报）$$G_t$$来表示Rewards的累积结果：

$$
G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\ldots=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}
$$

其中，$$\gamma$$为[0,1]的衰减因子，表示对长期reward的权重。

$$G_t$$可以表示为迭代的形式（递推公式）：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\ldots\\
=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\gamma^2R_{t+4}+\ldots)\\
=R_{t+1}+\gamma G_{t+1}
$$



有两点需要特别注意：

* $$G_t$$表示Agent在t时刻某状态s（在$$G_t$$中并没有体现状态s）的Return（回报）。也就是说，$$G_t$$是和时刻t和状态s相关的。不同的状态s和不同的时刻t回报是不同的。要计算$$G_t$$，首先要确定哪一个状态s作为起始节点。
* 式中$$R_{t+1}$$其实是当前时刻t的immediately reward，并非t+1时刻的reward。这是作者的约定，也可以将式中的$$R_{t+1}$$换成$$R_t$$，但是需要将整本书的符号都推倒重来一遍。还不太清楚作者这样约定的原因，能带来什么好处？难道是数学上的计算便利？

# 策略函数

策略函数$$\pi(a\mid s)$$是指Agent在状态s下选取动作a的概率，因此$$\pi(a\mid s)$$是一个2元的概率函数，表示从状态s到动作a的映射概率。

$$\pi(a\mid s)$$是强化学习中需要优化的参数之一：在迭代优化过程中，需要不断调整$$\pi(a\mid s)$$使得最终的结果胜率最高。

# status value function和action value function

$$v_{\pi}(s)$$表达了状态s在策略$$\pi$$下的好坏（**状态价值函数**），可以使用return的期望来衡量：

$$
v_{\pi}(s)=\mathbb{E_{\pi}}[G_t\mid S_t=s]=\mathbb{E_{\pi}}\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s\right]
$$

如下面的backup diagram所示：

![value-function-backup-diagram](/images/rl/value-function-backup-diagram.png)

假设当前处于状态s，即$$S_t=s$$。在策略$$\pi$$的作用下，从状态s可以有三个动作可以选择，每个动作进入Environment后会在2个后续状态s'及其对应的reward：r，于是我们评价状态s的价值，就需要将最后一层的所有reward进行加权平均（求期望），因为最后一层所有的reward都是状态s的可能结果。这是一个递归的过程，即最后一层的状态s'还会有相应的动作a'，接着是新的状态s''......


因此，$$v_{\pi}(s)$$中的k表示计算第几层的return加权平均（通过student mdp的例子说明计算过程）

$$q_{\pi}(s,,a)$$表达了状态s在策略$$\pi$$下，采取动作a的好坏（**动作价值函数**）：

$$
q_{\pi}(s,a)=\mathbb{E_{\pi}}[G_t\mid S_t=s,A_t=a]=\mathbb{E_{\pi}}\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t=s,A_t=a\right]
$$

下图有助于理解$$q_{\pi}(s,a)$$，见图中橙色线条标出部分。

![q-function-backup-diagram](/images/rl/q-function-backup-diagram.png)

将$$q_{\pi}(a,s)$$进一步展开可得：
$$
q_{\pi}(a,s)=\sum_{r}\sum_{s'}p(s',r\mid a,s)[r+\gamma\mathbb{E_{\pi}}[G_{t+1}\mid S_{t+1}=s']\\
=\sum_{r,s'}p(s',r\mid a,s)[r+\gamma v_{\pi}(s')]
$$
这是$$q_{\pi}(a,s)$$和$$v_{\pi}(s')$$的关系表达式，更能说明$$q_{\pi}(a,s)$$的计算过程：对于已知的<a,s>，Environment计算动作a的价值时是一个迭代的过程。比如在上图中，从节点a出发的下一个时刻的状态s'有两个，不妨记做$$s_{1}^{'}，s_{2}^{'}$$，其对应的reward分别为$$r_1$$和$$r_2$$，则$$q_{\pi}(a,s)$$的计算如下：
$$
q_{\pi}(a,s)=p(s_{1}^{'},r_1\mid a,s)[r_1+\gamma v_{\pi}(s_{1}^{'})]+p(s_{2}^{'},r_2\mid a,s)[r_2+\gamma v_{\pi}(s_{2}^{'})]
$$


从上图也可以看出$$q_{\pi}$$和$$v_{\pi}$$的关系：
$$
v_{\pi}(s)=\sum_{a}\pi(a\mid s)q_{\pi}(s,a)
$$

将$$q_{\pi}(s,a)$$的展开式代入可得：

$$
v_{\pi}(s)=\sum_{a}\pi(a\mid s)\sum_{r,s'}p(s',r\mid a,s)[r+\gamma v_{\pi}(s')]
$$

这就是状态价值函数的Bellman等式，即迭代公式。作者在这里有一段非常精彩的论述，摘录如下：

> where it is implicit that the actions, a, are taken from the set $$A(s)$$, that the next states, $$s'$$ , are taken from the set $$S$$ (or from $$S+$$ in the case of an episodic problem), and thatt he rewards, $$r$$, are taken from the set $$R$$. Note also how in the last equation we havem erged the two sums, one over all the values of $$s'$$  and the other over all the values of $$r$$, into one sum over all the possible values of both. We use this kind of merged sum often to simplify formulas. Note how the final expression can be read easily as an expected value. It is really a sum over all values of the three variables, $$a$$, $$s'$$ , and $$r$$. For each triple, we compute its probability, $$\pi(a\mid s)p(s' , r \mid s, a)$$, weight the quantity in brackets by that probability, then sum over all possibilities to get an expected value.

# 案例分析

> 画出student mdp的状态转移图和backup diagram，计算state value & action value，最终给出矩阵的计算形式。

下面是一个有趣的例子（替换了原作中的Facebook为Wechat），其中的圆圈代表了“状态”，方框代表了“结束状态”，每条线段（弧线）标明了状态转移的方向和概率。假设这门课只需要三节课就可结业，那么下图表达了同学们在学习过程中的各种状态及其转移概率。比如，上第一节课的时候，你有50%的概率顺利进入了第二节课，但是也有50%的概率经受不住微信的诱惑开始不停的刷。而刷微信容易上瘾，即90%的情况下你会不停的刷，直到某个时刻（10%的概率）重新回到第一节课。在第三节课，可能有40%的概率觉得差不多了，就到Pub喝点小酒庆祝......

当然，这个例子的状态图并没有完整描述上课的所有状态序列，这只是其中的一种可能状态序列。

注意到，每个状态节点的出线概率之和一定为1。比如，第二节课有两条出线，一条指向S节点（概率为0.2），一条指向C3节点（概率为0.8）。这很容易理解：有几条出线代表了从当前状态存在几种转移到其他状态的路径。

![makov-chains-student-1](images/rl/student-mdp-orig.png)

由此，概率转移矩阵为（矩阵的列顺序为C1C2C3PpassPubWechatSleep，如果能够标出矩阵行和列的label能够更清楚的说明问题，可惜在markdown里面不知道如何操作）：

$$
P=\left[\begin{matrix}
&0.5&&&&&0.5\\
&&0.8&&&0.2&\\
&&&0.4&0.6&&\\
0.2&0.4&0.4&&&&\\
&&&&&1.0&\\
&&&&&&\\
0.1&&&&&&0.9\\

\end{matrix}
\right]
$$

于是，下面的采样序列都是合理的，也可以计算这样的采样序列的达成概率。

* C1C2C3PassSleep：非常认真的学生
* C1WechatWwchatC1C2Sleep：第一节课走神的学生
* C1C2C3PubC1WechatWechatWechatC1C2C3PubC3PassSleep：经常走神的学生

## 计算状态价值函数

下面给状态转移图加上reward，即每个状态的immediately reward，参见下图：

![student-mdp-reward](/home/subaochen/git/subaochen.github.io/images/rl/student-mdp-reward.png)

下面以$$\gamma=1$$为例，计算节点C3的价值，如下图所示，节点C3的价值和C3的即时奖励以及下一节点的价值有关（根据状态价值函数的递推公式）。显然，Pass节点的价值为10。

![student-mdp-reward-c3](/home/subaochen/git/subaochen.github.io/images/rl/student-mdp-reward-c3.png)

上图中，计算C3节点的价值时使用到了Pub节点的价值，那么Pub节点的价值0.8是如何计算出来的呢？实际上，假设C1，C2，C3，Pass，Pub，Wechat节点的价值分别为v1,v2,v3,v4,v5,v6，我们可以列出如下的方程：
$$
v1=-2+0.5*v2+0.5*v6\\
v2=-2+0.8*v3\\
v3=-2+0.6*v4+0.4*v5\\
v4=10\\
v5=1+0.2*v1+0.4*v2+0.4*v3\\
v6=-1+0.1*v1+0.9*v6
$$


