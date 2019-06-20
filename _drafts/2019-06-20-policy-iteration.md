策略优化，或者寻找最优策略的三个步骤是：

1. 随机选择一个策略，比如uniform random policy，计算每个状态的值函数。这个随机选择的策略不一定（一般来说不是）最优策略，在这里计算状态的价值函数的目的是为了下一步policy improvement提供评判的**基准**。
2. 在第一步的基础上，在每一个状态都采取一个新的策略使得新策略下状态的价值函数都大于老策略下的状态价值函数。通常这个新的策略是greedy policy。
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

当采用greedy策略时，权重$$\pi(a\mid s)$$会向greedy策略选中的动作集中。假设在greedy策略下状态s的动作空间的维度为m，$$m\ge 1$$，则有$$\frac{1}{m}\times q_{\pi^{'}}(s,\pi^{'}(s))=v_{\pi}(s)$$（如果m>1则意味着greedy策略找到了m个动作都使得$$q_{\pi^{'}}(s,a)$$取得最大值，显然此时这m个动作的$$q_{\pi^{'}}(s,a)$$是相等的），因此显然有：$$q_{\pi^{'}}(s,\pi^{'}(s))\ge v_{\pi}(s)$$。下面逐步展开$$q_{\pi^{'}}(s,\pi^{'}(s))$$即可证明在greedy策略下，$$v_{\pi^{'}}\ge v_{\pi}(s)$$：
$$
\begin{align}
v_{\pi}(s)&\le q_{\pi^{'}}(s,\pi^{'}(s))\\
&=\mathbb{E_{\pi}}\left[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_t=t,A_t=\pi^{'}(s)\right]\\
&=
\end{align}
$$


**结论**：greedy策略可以实现policy improvement。有没有其他策略也可以实现policy improvement呢？作者没有阐述，有待进一步观察和研究。

#policy iteration

