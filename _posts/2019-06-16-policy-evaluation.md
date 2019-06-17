策略评估（policy evaluation）是指如何评估给定的策略$$\pi$$？


$$
v_{k+1}(s)=\sum_{a\in\mathcal{A}}\pi(a\mid s)\left(R^a_{s}+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}_{ss'}^av_{k}(s')\right)
$$
如何理解这个递推公式？通过下面的实例可以看的更清楚。

# 策略评估案例分析

