# 最优策略（optimal policy）

如何衡量策略的好坏呢？作者的定义为：

*如果策略$$\pi$$使得其状态的回报（return）的期望大于或者等于其他任意的策略$$\pi^{'}$$，则称策略$$\pi$$为最优策略。*

如下图所示，对于给定的状态s，存在不同的策略$$\pi$$，使得从状态s到动作a的路径不同。不同的路径回报不同，能够使得状态s的回报最大的策略$$\pi$$即为最优策略。

![optimal-policy-value-function](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/optimal-policy-value-function.png)

根据$$v(s)$$的定义，$$v(s)$$即回报的期望，因此寻找最优策略的问题转化为探索不同的$$\pi$$，找到使得$$v(s)$$最大的$$\pi$$，记此时的状态价值函数为$$v_{*}(s)$$，即：

$$
v_{*}(s)=\max_{\pi}v_{\pi}(s)
$$

直观的理解，如果假设动作a的价值是固定的，则不同的$$\pi$$使得从状态s出发的a的概率不同，即a的权重不同，因此导致$$v(s)$$不同。能够最大化$$v(s)$$的$$\pi$$即为最优策略，记为$$\pi_{*}$$。

同理，对于动作价值函数$$q(s,a)$$也有类似的定义：

$$
q_{*}(a,s)=\max_{\pi}q_{\pi}(s,a)
$$

即，策略$$\pi_{*}$$使得在状态s下采取动作a时其动作价值函数取最大值。

$$
\begin{align}
q_{*}(a,s)
&=\max_{\pi}q_\pi(s,a)\\
&=\max_\pi\mathbb{E_\pi}[R_{t+1}+\gamma G_{t+1}\mid S_t=s,A_t=a]\\
&=\mathbb{E_{\pi}}[R_{t+1}+\gamma G_{t+1}\mid S_t=t,A_t=a]\\
&=\mathbb{E}[R_{t+1}+\gamma v_{*}(S_{t+1})\mid S_t=t,A_t=a]
\end{align}
$$



那么，哪一条路径是最优的路径呢？无论是$$v(s)$$还是$$q(a,s)$$都是指特定节点的价值，那么如何根据节点的价值，计算出采取怎样的策略，获得最优的路径呢？这个问题可以表述为：
$$
v_{\pi}^{*}(s)=\max_{a}q^{*}_{\pi}(a,s)
$$

# 最优策略的案例

还是student MDP问题。