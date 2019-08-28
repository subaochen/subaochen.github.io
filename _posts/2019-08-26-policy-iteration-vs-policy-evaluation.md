---
title: policy iteration释疑一则
type: post
categories:
- Reinforcement Learning
layout: post
date: 2019-08-26
tags: [deeplearning,reinforcement learning,policy interation]
status: publish
published: true
use_math: true
comments: true
---

# 问题的提出

将policy evaluation和policy iteration的算法图并列在一起，可以看出，在policy evalution中，计算状态价值函数是要考虑不同动作的收益和的，但是在policy iteration中的policy evaluation中，计算状态价值函数并没有考虑不同动作的收益和，看起来只是计算了$$q(s,a)$$作为状态价值函数，是什么道理呢？![policy evaluation vs policy iteration](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/images/rl/dp/policy-iteration-vs-policy-evaluation.svg.png)

# 释疑

独立的policy evaluation算法只是对策略的评估，并没有后续的优化的策略步骤，因此在评估一个策略对状态价值的影响时，需要考虑该策略下所有动作对状态的影响，这是必然的。独立的策略评估很少单独使用，原因见下面的策略迭代（policy iteration）的分析。

策略迭代的第一步是策略评估，但是这里的策略评估和独立的策略评估过程是不同的：策略迭代中的策略评估，其实是在一轮策略增强的基础上进行评估的，也就是说，评估上一轮策略增强的效果。如下图所示：

![](https://raw.githubusercontent.com/subaochen/subaochen.github.io/master/images/rl/dp/policy-iteration-cycle.png)

在整个策略序列$$\pi_0,\pi_1, \pi_2\ldots$$中，除了$$\pi_0$$外，$$\pi_1, \pi_2\ldots$$都是策略增强的结果，即在策略评估的基础上，往前看一步，选择使得$$q(s,a)$$最大化的动作$$a$$。也就是说，$$\pi_1, \pi_2\ldots$$都是局部最优策略。虽然$$\pi_0$$不是策略增强的结果，但是如果我们把初始的策略也看做是一个局部最优策略：毕竟，在$$\pi_0$$的前面并没有参照，一个唯一的存在自然可以看做是当前最优的。于是我们可以说，$$\pi_0,\pi_1, \pi_2\ldots$$都是局部最优策略。对于局部最优策略的评估，显然策略已经给出了使得当前的$$q(s,a)$$最大化的动作$$a$$（科可能不止一个，但是效果一样），因此我们在评估这个局部最优策略时，只需要评估这个最优动作$$a$$带来的价值即可，无需考虑所有的动作，**这就是在策略迭代的策略评估中没有$$\sum_a\pi(a\mid s)$$这一项的原因**。

简单的说，策略迭代中的策略评估，是评估逐步优化（比如greedy policy）的策略，而非任意的策略（独立的策略评估针对任意的策略），因此只需考虑最优动作的价值。下面是一个实际的案例（摘自Jack's car rental）：

```python
        # policy evaluation (in-place)
        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_state_value = expected_return([i, j], policy[i, j],
                        value, constant_returned_cars, POISSON_UPPER_BOUND)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break
```

可以看出，策略评估过程中，每一个状态只调用一次expected_return方法计算状态的价值：因为给出的策略（policy[i,j]）已经是局部最优了，只需要计算这个局部最优的策略（动作）的价值即可。