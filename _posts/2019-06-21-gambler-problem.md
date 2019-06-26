---
title: DP学习笔记-使用值迭代解决赌徒问题
type: post
categories:
- deeplearning
layout: post
date: 2019-06-21
tags: [deeplearning,dynamic programming,reinforcement learning]
status: publish
published: true
use_math: true
comments: true
---

# 赌徒问题

假设赌徒根据硬币的正反面来决定输赢。如果硬币正面（head）朝上，赌徒可以拿走和赌注等量的筹码，如果硬币反面朝上，赌徒则输掉赌注。当赌徒赢够100元或者输掉全部赌本则赌局结束。求（硬币正面朝上）在不同的概率和不同的赌资下，赌徒赢够100元的概率（或许还应该加上赌博轮次的限制？比如只抛100次硬币）。

# 问题分析

显然，不同的赌资应该采取不同的策略。比如，如果赌徒只有1块钱，只有全部压上（赌注）才有机会最终赢得100块。但是随着赌资的增加，赌徒必须考虑要拿出多少钱作为赌注。比如，如果赌徒手中已经有51块前，显然不能将51块全部压上：如果赢了，则无法实现赢够100元的目标（超出了），如果输掉，则变得一穷二白。因此，全部压上51块钱是一个糟糕的策略。

假设问题的状态空间s为{1,2,...,99}，即赌徒手中现有的赌资（capital）。问题的动作空间a为{0,1,2,...,min(s,100-s)}，即赌徒所下的赌注（stake），则问题转化为在不同的赌资条件下，赌徒采取何种（最优）下注策略，能够尽快赢够100元？这是一个策略优化问题，可以采用价值迭代的方式来寻找最优策略。

# 代码实现
主要参考了：https://github.com/dennybritz/reinforcement-learning 。代码的编写是直线型的（不是面向对象），因此还是比较容易理解的。其中主要的是两点：

* 值迭代的过程：首先使用贪心算法计算每个状态的价值函数，直到价值函数收敛，然后再**一次性**计算最优策略。
* 对于sutton book中公式4.10的理解很重要：基于贪心算法，当前状态的价值函数是其对应的最大的动作价值函数（以当前赌资为98元为例），再次列出如下：

$$
v_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_k(s')]
$$

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/gambler-problem-98.png)


```Python
import numpy as np
import matplotlib.pyplot as plt


def value_iteration_for_gamblers(p_h, theta=0.0001, gamma=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
        theta: 迭代结束条件
        gamma: 衰减因子
    """
    # The reward is zero on all transitions except those on which the gambler reaches his goal,
    # when it is +1.
    R = np.zeros(101)
    R[100] = 1

    # We introduce two dummy states corresponding to termination with capital of 0 and 100
    V = np.zeros(101)

    def one_step_lookahead(s, V, R):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            s: The gambler’s capital. Integer.当前的状态
            V: The vector that contains values at each state.
            R: The reward vector.

        Returns:
            A vector containing the expected value of each action.
            Its length equals to the number of actions.
        """
        A = np.zeros(101)
        stakes = range(1, min(s, 100 - s) + 1)  # Your minimum bet is 1, maximum bet is min(s, 100-s).
        for a in stakes:
            # R[s+a], R[s-a] are immediate rewards.
            # V[s+a], V[s-a] are values of the next states.
            # This is the core of the Bellman equation: The expected value of your action is
            # the sum of immediate rewards and the value of the next state.
            A[a] = p_h * (R[s + a] + V[s + a] * gamma) + (1 - p_h) * (
                    R[s - a] + V[s - a] * gamma)
        return A

    # 第几次重新设置状态价值函数？`
    sweep = 0
    while True:
        print("=============sweep=" + str(sweep) + "==============")
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(1, 100):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, R)
            # print(s,A,V) if you want to debug.
            # 大量的调试输出
            print("s=")
            print(s)
            print("A=")
            print(A)
            print("V=")
            print(V)
            print("R=")
            print(R)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value

        sweep = sweep + 1

        # 画出每次迭代的状态价值函数曲线，观察状态价值函数的变化趋势
        plt.plot(range(100), V[:100])

        # Check if we can stop
        if delta < theta:
            plt.xlabel('Capital')
            plt.ylabel('Value Estimates')
            plt.title('Final Policy(action stakes) vs. State(Capital),p_h=' + str(p_h))
            plt.show()

            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros(100)
    for s in range(1, 100):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, R)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s] = best_action

    return policy, V


def draw_value_estimates(p_h):
    x = range(100)
    y = v[:100]
    plt.plot(x, y)
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.title('Final Policy (action stake) vs State (Capital),p_h=' + str(p_h))
    plt.show()


def draw_policy(p_h):
    x = range(100)
    y = policy
    plt.bar(x, y, align='center', alpha=0.5)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.title('Capital vs Final Policy(ph=' + str(p_h) + ')')
    plt.show()


if __name__ == '__main__':
    for p_h in (0.4, 0.55):
        policy, v = value_iteration_for_gamblers(p_h)

        print("Optimized Policy(p_h=" + str(p_h) + "):")
        print(policy)
        print("")

        print("Optimized Value Function(p_h=" + str(p_h) + "):")
        print(v)
        print("")

        # draw_value_estimates(p_h)
        draw_policy(p_h)
```

当p_h=0.25的时候，状态价值函数的计算过程如下图所示：

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/gambler-problem-ph-025-value-function.png)

最优策略如下图所示：

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/gambler-problem-ph-025-policy.png)

当p_h=0.55的时候，状态价值函数的计算过程如下图所示：

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/gambler-problem-ph-055-value-function.png)

最优策略如下图所示：

![](https://github.com/subaochen/subaochen.github.io/raw/master/images/rl/dp/gambler-problem-ph-055-policy.png)

可以看出，当硬币正面概率为0.55的时候，最优策略几乎总是下注1块，有点匪夷所思，难道计算错了？