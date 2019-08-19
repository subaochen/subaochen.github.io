---
title: MDP学习笔记-最优价值函数和最优策略
type: post
categories:
- deeplearning
layout: post
date: 2019-08-19
tags: [deeplearning,mdp,optimal value function,optimal policy,reinforcement learning,grid world]
status: publish
published: true
use_math: true
comments: true
---

# 最优策略（optimal policy）

面对一个问题，可能有策略千万条，哪一条是最好的呢？事实上，总是存在：

> *如果策略$$\pi$$使得其状态的回报（return）的期望大于或者等于其他任意的策略$$\pi^{'}$$，则称策略$$\pi$$为最优策略。*

状态的回报的期望即为$$v_{\pi}(s)$$，即：$$\mathbb{E}_{\pi}[G_t|S_t=s,A_t=a]$$。也就是说，对于任意状态$$s\in\mathcal{S}$$都有$$v_{\pi}(s)>=v_{\pi^{'}}(s)$$，则称$$v_{\pi}(s)$$为最优状态价值函数，记为$$v_*(s)$$。显然有：
$$
v_*(s)=\max_{\pi}v_{\pi}(s), \forall s\in\mathcal{S}
$$
如下图所示，对于给定的状态s，存在不同的策略$$\pi$$，使得从状态s到动作a的路径不同。不同的路径回报不同，能够使得状态s的回报最大的策略$$\pi$$即为最优策略，记为$$\pi_*$$。

![optimal-policy-value-function](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/mdp/optimal-policy-value-function.png)

直观的理解，不同的动作方向会导向不同的状态$$s'$$，带来不同的状态价值（$$s'$$状态价值的加权平均，即期望）。能够使得$$s'$$的状态价值最大化的动作记为最优的动作，或者说是最优策略。也就是说，通过状态价值函数判断最优策略，需要计算下一级节点的状态价值的期望，这就是作者所说的“one-step search”，只需要往前走一步，即可以判断当前的动作组合中，哪个是最优的：能够使得$$s'$$的状态价值函数的期望最大化的动作，选择这个最优的动作即为最佳策略。显然，这是一种完全的“贪心算法”，只基于下一步做出的决策。幸运的是，$$s'$$的状态价值函数实际上已经包含了长期的价值（思考一下状态价值函数的定义），因此：

> A one-step-ahead search yields the long-term optimal actions.

同理，对于动作价值函数$$q(s,a)$$也有类似的定义：
$$
q_{*}(s,a)=\max_{\pi}q_{\pi}(s,a)
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

# 最优策略的案例：grid world问题

grid world问题参见：在那里，仅仅给出了在每个格子的状态价值函数（策略$$\pi$$为随机平均策略）。下面的方法可以计算grid world的最优策略：

```python
def grid_world_optimal_policy():
    """计算格子世界的最优价值函数和最优策略
    """
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    # 通过一个数组来表示每一个格子的最优动作，1表示在相应的方向上最优的
    optimal_policy = np.zeros((WORLD_SIZE, WORLD_SIZE, len(ACTIONS)))
    episode = 0
    while True:
        episode = episode + 1
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # 保存当前格子所有action下的state value
                action_values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    action_values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(action_values)
                optimal_policy[i, j] = get_optimal_actions(action_values)
        error = np.sum(np.abs(new_value - value))
        if error < 1e-4:
            break
        # 观察每一轮次状态价值函数及其误差的变化情况
        print(f"{episode}-{np.round(error,decimals=5)}:\n{np.round(new_value,decimals=2)}")
        value = new_value
    print(f"optimal policy:{optimal_policy}")


def get_optimal_actions(values):
    """计算当前轮次格子的最优动作
    :param values:格子的状态价值
    :return: 当前的最优动作。解读这个最优动作数组，要参考ACTIONS中四个动作的方向定义，
    数值为1表示此动作为最优动作
    """
    optimal_actions = np.zeros(len(ACTIONS))
    indices = np.where(values == np.amax(values))
    for index in indices[0]:
        optimal_actions[index] = 1
    return optimal_actions
```

完整的grid world程序参见：[grid world源码](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/resources/grid_world.py)